using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
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

namespace OnnxStack.StableDiffusion.Diffusers.LatentConsistency
{
    public class ControlNetDiffuser : LatentConsistencyDiffuser
    {
        protected ControlNetModel _controlNet;

        /// <summary>
        /// Initializes a new instance of the <see cref="ControlNetDiffuser"/> class.
        /// </summary>
        /// <param name="controlNet">The control net.</param>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public ControlNetDiffuser(ControlNetModel controlNet, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, MemoryModeType memoryMode, ILogger logger = default)
            : base(unet, vaeDecoder, vaeEncoder, memoryMode, logger)
        {
            _controlNet = controlNet;
        }

        /// <summary>
        /// Gets the type of the diffuser.
        /// </summary>
        public override DiffuserType DiffuserType => DiffuserType.ControlNet;


        /// <summary>
        /// Runs the scheduler steps.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <param name="performGuidance">if set to <c>true</c> [perform guidance].</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        public override async Task<DenseTensor<float>> DiffuseAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions, PromptEmbeddingsResult promptEmbeddings, bool performGuidance, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            // Get Scheduler
            using (var scheduler = GetScheduler(schedulerOptions))
            {
                // Get timesteps
                var timesteps = GetTimesteps(schedulerOptions, scheduler);

                // Create latent sample
                var latents = await PrepareLatentsAsync(promptOptions, schedulerOptions, scheduler, timesteps);

                // Get Guidance Scale Embedding
                var guidanceEmbeddings = GetGuidanceScaleEmbedding(schedulerOptions.GuidanceScale);

                // Get Model metadata
                var metadata = await _unet.GetMetadataAsync();

                // Get Model metadata
                var controlNetMetadata = await _controlNet.GetMetadataAsync();

                // Control Image
                var controlImageTensor = await PrepareControlImage(promptOptions, schedulerOptions);

                // Denoised result
                DenseTensor<float> denoised = null;

                // Loop though the timesteps
                var step = 0;
                foreach (var timestep in timesteps)
                {
                    step++;
                    var stepTime = Stopwatch.GetTimestamp();
                    cancellationToken.ThrowIfCancellationRequested();

                    // Create input tensor.
                    var inputTensor = scheduler.ScaleInput(latents, timestep);
                    var timestepTensor = CreateTimestepTensor(timestep);
                    var conditioningScale = CreateConditioningScaleTensor(schedulerOptions.ConditioningScale);

                    var batchCount = 1;
                    var outputDimension = schedulerOptions.GetScaledDimension(batchCount);
                    using (var inferenceParameters = new OnnxInferenceParameters(metadata))
                    {
                        inferenceParameters.AddInputTensor(inputTensor);
                        inferenceParameters.AddInputTensor(timestepTensor);
                        inferenceParameters.AddInputTensor(promptEmbeddings.PromptEmbeds);
                        inferenceParameters.AddInputTensor(guidanceEmbeddings);

                        // ControlNet
                        using (var controlNetParameters = new OnnxInferenceParameters(controlNetMetadata))
                        {
                            controlNetParameters.AddInputTensor(inputTensor);
                            controlNetParameters.AddInputTensor(timestepTensor);
                            controlNetParameters.AddInputTensor(promptEmbeddings.PromptEmbeds);
                            controlNetParameters.AddInputTensor(guidanceEmbeddings);
                            controlNetParameters.AddInputTensor(controlImageTensor);
                            if (controlNetMetadata.Inputs.Count == 5)
                                controlNetParameters.AddInputTensor(conditioningScale);

                            // Optimization: Pre-allocate device buffers for inputs
                            foreach (var item in controlNetMetadata.Outputs)
                                controlNetParameters.AddOutputBuffer();

                            // ControlNet inference
                            var controlNetResults = _controlNet.RunInference(controlNetParameters);

                            // Add ControlNet outputs to Unet input
                            foreach (var item in controlNetResults)
                                inferenceParameters.AddInput(item);

                            // Add output buffer
                            inferenceParameters.AddOutputBuffer(outputDimension);

                            // Unet inference
                            var results = await _unet.RunInferenceAsync(inferenceParameters);
                            using (var result = results.First())
                            {
                                var noisePred = result.ToDenseTensor();

                                // Scheduler Step
                                var schedulerResult = scheduler.Step(noisePred, timestep, latents);

                                latents = schedulerResult.Result;
                                denoised = schedulerResult.SampleData;
                            }
                        }
                    }

                    ReportProgress(progressCallback, step, timesteps.Count, latents);
                    _logger?.LogEnd($"Step {step}/{timesteps.Count}", stepTime);
                }

                // Unload if required
                if (_memoryMode != MemoryModeType.Maximum)
                {
                    await _unet.UnloadAsync();
                    await _controlNet.UnloadAsync();
                }

                // Decode Latents
                return await DecodeLatentsAsync(promptOptions, schedulerOptions, latents);
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
        protected override Task<DenseTensor<float>> PrepareLatentsAsync(PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps)
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
        protected async Task<DenseTensor<float>> PrepareControlImage(PromptOptions promptOptions, SchedulerOptions schedulerOptions)
        {
            return await promptOptions.InputContolImage.ToDenseTensorAsync(schedulerOptions.Height, schedulerOptions.Width , ImageNormalizeType.ZeroToOne);
        }
    }
}
