using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers.StableDiffusion
{
    public sealed class InstructControlNetDiffuser : InstructDiffuser
    {
        private ControlNetModel _controlNet;

        /// <summary>
        /// Initializes a new instance of the <see cref="InstructControlNetDiffuser"/> class.
        /// </summary>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public InstructControlNetDiffuser(ControlNetModel controlNet, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, ILogger logger = default)
            : base(unet, vaeDecoder, vaeEncoder, logger)
        {
            _controlNet = controlNet;
        }


        /// <summary>
        /// Gets the type of the diffuser.
        /// </summary>
        public override DiffuserType DiffuserType => DiffuserType.ControlNetImage;


        /// <summary>
        /// Gets the timesteps.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <returns></returns>
        protected override IReadOnlyList<int> GetTimesteps(SchedulerOptions options, IScheduler scheduler)
        {
            if (!options.Timesteps.IsNullOrEmpty())
                return options.Timesteps;

            return scheduler.Timesteps;
        }


        /// <summary>
        /// Runs the scheduler steps.
        /// </summary>
        /// <param name="options"></param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async Task<DenseTensor<float>> DiffuseAsync(DiffuseOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var generateOptions = options.GenerateOptions;
            var schedulerOptions = generateOptions.SchedulerOptions;
            var performGuidance = ShouldPerformGuidance(schedulerOptions);
            var promptEmbeds = options.PromptEmbeddings.GetPromptEmbeds(performGuidance);
            var pooledPromptEmbeds = options.PromptEmbeddings.GetPooledPromptEmbeds(performGuidance);
            using (var scheduler = GetScheduler(schedulerOptions))
            {
                // Get Unet Model metadata
                var unetMetadata = await _unet.LoadAsync(cancellationToken: cancellationToken);

                // Get ControlNet Model metadata
                var controlNetMetadata = await _controlNet.LoadAsync(cancellationToken: cancellationToken);

                // Get timesteps
                var timesteps = GetTimesteps(schedulerOptions, scheduler);

                // Create Noise Latents
                progressCallback.Notify("Prepare Input...");
                var latents = await PrepareLatentsAsync(generateOptions, scheduler, timesteps, cancellationToken);

                // Create Image Latents
                var imageLatents = await PrepareImageLatentsAsync(generateOptions, scheduler, timesteps, cancellationToken);

                // Create Control Image
                var controlImage = await PrepareControlImage(generateOptions);

                // Loop though the timesteps
                var step = 0;
                ReportProgress(progressCallback, "Step", 0, timesteps.Count, 0);
                foreach (var timestep in timesteps)
                {
                    step++;
                    var stepTime = Stopwatch.GetTimestamp();
                    cancellationToken.ThrowIfCancellationRequested();

                    // Prepare latents
                    var batchCount = performGuidance ? 3 : 1;
                    var inputLatents = performGuidance ? latents.Repeat(batchCount) : latents;
                    var scaledLatents = scheduler.ScaleInput(inputLatents, timestep);

                    // Create input tensors.
                    var inputTensor = scaledLatents.Concatenate(imageLatents, 1);
                    var promptTensor = PreparePromptEmbeds(promptEmbeds);
                    var timestepTensor = CreateTimestepTensor(timestep);
                    var controlTensor = performGuidance ? controlImage.Repeat(batchCount) : controlImage;
                    var conditioningScale = CreateConditioningScaleTensor(schedulerOptions.ConditioningScale);

                    var outputDimension = schedulerOptions.GetScaledDimension(batchCount);
                    using (var inferenceParameters = new OnnxInferenceParameters(unetMetadata, cancellationToken))
                    {
                        inferenceParameters.AddInputTensor(inputTensor);
                        inferenceParameters.AddInputTensor(timestepTensor);
                        inferenceParameters.AddInputTensor(promptTensor);
                        inferenceParameters.AddOutputBuffer(outputDimension);

                        // ControlNet Inference
                        using (var controlNetParameters = new OnnxInferenceParameters(controlNetMetadata, cancellationToken))
                        {
                            controlNetParameters.AddInputTensor(scaledLatents);
                            controlNetParameters.AddInputTensor(timestepTensor);
                            controlNetParameters.AddInputTensor(promptTensor);
                            controlNetParameters.AddInputTensor(controlTensor);
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


                            // Unet Inference
                            var results = await _unet.RunInferenceAsync(inferenceParameters);
                            using (var result = results.First())
                            {
                                var noisePred = result.ToDenseTensor();

                                // Perform guidance
                                if (performGuidance)
                                    noisePred = PerformGuidance(noisePred, schedulerOptions.GuidanceScale, schedulerOptions.Strength);

                                // Scheduler Step
                                latents = scheduler.Step(noisePred, timestep, latents).Result;
                            }
                        }

                    }

                    ReportProgress(progressCallback, "Step", step, timesteps.Count, stepTime, latents);
                    _logger?.LogEnd(LogLevel.Debug, $"Step {step}/{timesteps.Count}", stepTime);
                }

                // Unload if required
                if (generateOptions.IsLowMemoryComputeEnabled)
                    await _unet.UnloadAsync();


                // Decode Latents
                return await DecodeLatentsAsync(generateOptions, latents, cancellationToken);
            }
        }



        /// <summary>
        /// Creates the Conditioning Scale tensor.
        /// </summary>
        /// <param name="conditioningScale">The conditioningScale.</param>
        /// <returns></returns>
        private static DenseTensor<double> CreateConditioningScaleTensor(float conditioningScale)
        {
            return new DenseTensor<double>(new double[] { conditioningScale }, new int[] { 1 });
        }


        /// <summary>
        /// Prepares the control image.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns></returns>
        private async Task<DenseTensor<float>> PrepareControlImage(GenerateOptions options)
        {
            var controlImageTensor = await options.InputContolImage.GetImageTensorAsync(options.SchedulerOptions.Height, options.SchedulerOptions.Width, ImageNormalizeType.ZeroToOne);
            if (_controlNet.InvertInput)
                InvertInputTensor(controlImageTensor);

            return controlImageTensor;
        }


        /// <summary>
        /// Inverts the input tensor.
        /// </summary>
        /// <param name="values">The values.</param>
        private static void InvertInputTensor(DenseTensor<float> values)
        {
            for (int j = 0; j < values.Length; j++)
            {
                values.SetValue(j, 1f - values.GetValue(j));
            }
        }

    }
}
