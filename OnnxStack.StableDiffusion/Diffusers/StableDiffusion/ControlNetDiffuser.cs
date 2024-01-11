using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers.StableDiffusion
{
    public class ControlNetDiffuser : StableDiffusionDiffuser
    {
        private readonly IControlNetImageService _controlNetImageService;

        /// <summary>
        /// Initializes a new instance of the <see cref="ControlNetDiffuser"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="onnxModelService">The onnx model service.</param>
        public ControlNetDiffuser(IOnnxModelService onnxModelService, IPromptService promptService, IControlNetImageService controlNetImageService, ILogger<ControlNetDiffuser> logger)
            : base(onnxModelService, promptService, logger)
        {
            _controlNetImageService = controlNetImageService;
        }


        /// <summary>
        /// Gets the type of the diffuser.
        /// </summary>
        public override DiffuserType DiffuserType => DiffuserType.ControlNet;


        /// <summary>
        /// Called on each Scheduler step.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <param name="performGuidance">if set to <c>true</c> [perform guidance].</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        protected override async Task<DenseTensor<float>> SchedulerStepAsync(ModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, PromptEmbeddingsResult promptEmbeddings, bool performGuidance, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            // Get Scheduler
            using (var scheduler = GetScheduler(schedulerOptions))
            {
                // Get timesteps
                var timesteps = GetTimesteps(schedulerOptions, scheduler);

                // Create latent sample
                var latents = await PrepareLatentsAsync(modelOptions, promptOptions, schedulerOptions, scheduler, timesteps);

                // Get Model metadata
                var metadata = _onnxModelService.GetModelMetadata(modelOptions.BaseModel, OnnxModelType.Unet);

                // Get Model metadata
                var controlNetMetadata = _onnxModelService.GetModelMetadata(modelOptions.ControlNetModel, OnnxModelType.ControlNet);

                // Control Image
                var controlImage = await PrepareControlImage(modelOptions, promptOptions, schedulerOptions);

                // Loop though the timesteps
                var step = 0;
                foreach (var timestep in timesteps)
                {
                    step++;
                    var stepTime = Stopwatch.GetTimestamp();
                    cancellationToken.ThrowIfCancellationRequested();

                    // Create input tensor.
                    var inputLatent = performGuidance ? latents.Repeat(2) : latents;
                    var inputTensor = scheduler.ScaleInput(inputLatent, timestep);
                    var timestepTensor = CreateTimestepTensor(timestep);
                    var controlImageTensor = performGuidance ? controlImage.Repeat(2) : controlImage;
                    var conditioningScale = CreateConditioningScaleTensor(schedulerOptions.ConditioningScale);

                    var outputChannels = performGuidance ? 2 : 1;
                    var outputDimension = schedulerOptions.GetScaledDimension(outputChannels);
                    using (var inferenceParameters = new OnnxInferenceParameters(metadata))
                    {
                        inferenceParameters.AddInputTensor(inputTensor);
                        inferenceParameters.AddInputTensor(timestepTensor);
                        inferenceParameters.AddInputTensor(promptEmbeddings.PromptEmbeds);

                        // ControlNet
                        using (var controlNetParameters = new OnnxInferenceParameters(controlNetMetadata))
                        {
                            controlNetParameters.AddInputTensor(inputTensor);
                            controlNetParameters.AddInputTensor(timestepTensor);
                            controlNetParameters.AddInputTensor(promptEmbeddings.PromptEmbeds);
                            controlNetParameters.AddInputTensor(controlImage);
                            if (controlNetMetadata.Inputs.Count == 5)
                                controlNetParameters.AddInputTensor(conditioningScale);

                            // Optimization: Pre-allocate device buffers for inputs
                            foreach (var item in controlNetMetadata.Outputs)
                                controlNetParameters.AddOutputBuffer();

                            // ControlNet inference
                            var controlNetResults = _onnxModelService.RunInference(modelOptions.ControlNetModel, OnnxModelType.ControlNet, controlNetParameters);

                            // Add ControlNet outputs to Unet input
                            foreach (var item in controlNetResults)
                                inferenceParameters.AddInput(item);

                            // Add output buffer
                            inferenceParameters.AddOutputBuffer(outputDimension);

                            // Unet inference
                            var results = await _onnxModelService.RunInferenceAsync(modelOptions.BaseModel, OnnxModelType.Unet, inferenceParameters);
                            using (var result = results.First())
                            {
                                var noisePred = result.ToDenseTensor();

                                // Perform guidance
                                if (performGuidance)
                                    noisePred = PerformGuidance(noisePred, schedulerOptions.GuidanceScale);

                                // Scheduler Step
                                latents = scheduler.Step(noisePred, timestep, latents).Result;
                            }
                        }
                    }

                    ReportProgress(progressCallback, step, timesteps.Count, latents);
                    _logger?.LogEnd($"Step {step}/{timesteps.Count}", stepTime);
                }

                // Decode Latents
                return await DecodeLatentsAsync(modelOptions, promptOptions, schedulerOptions, latents);
            }
        }


        /// <summary>
        /// Gets the timesteps.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <returns></returns>
        protected override IReadOnlyList<int> GetTimesteps(SchedulerOptions options, IScheduler scheduler)
        {
            return scheduler.Timesteps;
        }


        /// <summary>
        /// Prepares the input latents.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <returns></returns>
        protected override Task<DenseTensor<float>> PrepareLatentsAsync(ModelOptions model, PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps)
        {
            return Task.FromResult(scheduler.CreateRandomSample(options.GetScaledDimension(), scheduler.InitNoiseSigma));
        }


        /// <summary>
        /// Creates the Conditioning Scale tensor.
        /// </summary>
        /// <param name="conditioningScale">The conditioningScale.</param>
        /// <returns></returns>
        protected static DenseTensor<double> CreateConditioningScaleTensor(float conditioningScale)
        {
            return TensorHelper.CreateTensor(new double[] { conditioningScale }, new int[] { 1 });
        }


        /// <summary>
        /// Prepares the control image.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <returns></returns>
        protected async Task<DenseTensor<float>> PrepareControlImage(ModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions)
        {
            var controlImage = promptOptions.InputContolImage;
            if (schedulerOptions.IsControlImageProcessingEnabled)
            {
                controlImage = await _controlNetImageService.PrepareInputImage(modelOptions.ControlNetModel, promptOptions.InputContolImage, schedulerOptions.Height, schedulerOptions.Width);
            }
            return await controlImage.ToDenseTensorAsync(schedulerOptions.Height, schedulerOptions.Width , ImageNormalizeType.ZeroToOne);
        }
    }
}
