using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
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

namespace OnnxStack.StableDiffusion.Diffusers.StableDiffusionXL
{
    public class ControlNetDiffuser : StableDiffusionXLDiffuser
    {
        private readonly ControlNetModel _controlNet;

        /// <summary>
        /// Initializes a new instance of the <see cref="ControlNetDiffuser"/> class.
        /// </summary>
        /// <param name="controlNet">The control net.</param>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public ControlNetDiffuser(ControlNetModel controlNet, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, ILogger logger = default)
            : base(unet, vaeDecoder, vaeEncoder, logger)
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
        /// <param name="options"></param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async Task<DenseTensor<float>> DiffuseAsync(DiffuseOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var generateOptions = options.GenerateOptions;
            var schedulerOptions = generateOptions.SchedulerOptions;
            var performGuidance = ShouldPerformGuidance(schedulerOptions);
            var promptEmbedsCond = options.PromptEmbeddings.PromptEmbeds;
            var pooledPromptEmbedsCond = options.PromptEmbeddings.PooledPromptEmbeds;
            var promptEmbedsUncond = options.PromptEmbeddings.NegativePromptEmbeds;
            var pooledPromptEmbedsUncond = options.PromptEmbeddings.NegativePooledPromptEmbeds;

            using (var scheduler = GetScheduler(schedulerOptions))
            {
                // Get Model metadata
                var metadata = await _unet.LoadAsync(cancellationToken: cancellationToken);

                // Get Model metadata
                var controlNetMetadata = await _controlNet.LoadAsync(cancellationToken: cancellationToken);

                // Get timesteps
                var timesteps = GetTimesteps(schedulerOptions, scheduler);

                // Create latent sample
                progressCallback.Notify("Prepare Input...");
                var latents = await PrepareLatentsAsync(generateOptions, scheduler, timesteps, cancellationToken);

                // Get Time ids
                var addTimeIds = GetAddTimeIds(schedulerOptions);

                // Control Image
                var controlImage = await PrepareControlImage(generateOptions);

                // Loop though the timesteps
                var step = 0;
                ReportProgress(progressCallback, "Step", 0, timesteps.Count, 0);
                foreach (var timestep in timesteps)
                {
                    step++;
                    var stepTime = Stopwatch.GetTimestamp();
                    cancellationToken.ThrowIfCancellationRequested();

                    // Create input tensor.
                    var inputTensor = scheduler.ScaleInput(latents, timestep);
                    var timestepTensor = CreateTimestepTensor(timestep);
                    var conditioningScale = CreateConditioningScaleTensor(schedulerOptions.ConditioningScale);

                    var outputDimension = schedulerOptions.GetScaledDimension();
                    using (var unetCondParams = new OnnxInferenceParameters(metadata, cancellationToken))
                    using (var unetUncondParams = new OnnxInferenceParameters(metadata, cancellationToken))
                    using (var controlNetCondParams = new OnnxInferenceParameters(controlNetMetadata, cancellationToken))
                    using (var controlNetUncondParams = new OnnxInferenceParameters(controlNetMetadata, cancellationToken))
                    {
                        // UNet
                        unetCondParams.AddInputTensor(inputTensor);
                        unetCondParams.AddInputTensor(timestepTensor);
                        unetCondParams.AddInputTensor(promptEmbedsCond);
                        unetCondParams.AddInputTensor(pooledPromptEmbedsCond);
                        unetCondParams.AddInputTensor(addTimeIds);

                        // ControlNet
                        controlNetCondParams.AddInputTensor(inputTensor);
                        controlNetCondParams.AddInputTensor(timestepTensor);
                        controlNetCondParams.AddInputTensor(promptEmbedsCond);
                        controlNetCondParams.AddInputTensor(pooledPromptEmbedsCond);
                        controlNetCondParams.AddInputTensor(addTimeIds);
                        controlNetCondParams.AddInputTensor(controlImage);
                        controlNetCondParams.AddInputTensor(conditioningScale);

                        // Output
                        unetCondParams.AddOutputBuffer(outputDimension);
                        foreach (var item in controlNetMetadata.Outputs)
                            controlNetCondParams.AddOutputBuffer();

                        // Inference
                        var controlNetResults = _controlNet.RunInference(controlNetCondParams);
                        foreach (var item in controlNetResults)
                            unetCondParams.AddInput(item);
                        var unetCondResults = await _unet.RunInferenceAsync(unetCondParams);

                        // Unconditional
                        var unetUncondResults = default(IReadOnlyCollection<OrtValue>);
                        if (performGuidance)
                        {
                            // UNet
                            unetUncondParams.AddInputTensor(inputTensor);
                            unetUncondParams.AddInputTensor(timestepTensor);
                            unetUncondParams.AddInputTensor(promptEmbedsUncond);
                            unetUncondParams.AddInputTensor(pooledPromptEmbedsUncond);
                            unetUncondParams.AddInputTensor(addTimeIds);

                            // ControlNet
                            controlNetUncondParams.AddInputTensor(inputTensor);
                            controlNetUncondParams.AddInputTensor(timestepTensor);
                            controlNetUncondParams.AddInputTensor(promptEmbedsCond);
                            controlNetUncondParams.AddInputTensor(pooledPromptEmbedsCond);
                            controlNetUncondParams.AddInputTensor(addTimeIds);
                            controlNetUncondParams.AddInputTensor(controlImage);
                            controlNetUncondParams.AddInputTensor(conditioningScale);

                            // Output
                            unetUncondParams.AddOutputBuffer(outputDimension);
                            foreach (var item in controlNetMetadata.Outputs)
                                controlNetUncondParams.AddOutputBuffer();

                            // Inference
                            var controlNetUncondResults = _controlNet.RunInference(controlNetUncondParams);
                            foreach (var item in controlNetUncondResults)
                                unetUncondParams.AddInput(item);

                            unetUncondResults = await _unet.RunInferenceAsync(unetUncondParams);
                        }

                        // Result
                        using (var unetCondResult = unetCondResults.First())
                        using (var unetUncondResult = unetUncondResults?.FirstOrDefault())
                        {
                            var noisePred = unetCondResult.ToDenseTensor();

                            // Perform guidance
                            if (performGuidance)
                                noisePred = PerformGuidance(noisePred, unetUncondResult.ToDenseTensor(), schedulerOptions.GuidanceScale);

                            // Scheduler Step
                            latents = scheduler.Step(noisePred, timestep, latents).Result;
                        }
                    }

                    ReportProgress(progressCallback, "Step", step, timesteps.Count, stepTime, latents);
                    _logger?.LogEnd(LogLevel.Debug, $"Step {step}/{timesteps.Count}", stepTime);
                }

                // Unload if required
                if (generateOptions.IsLowMemoryComputeEnabled)
                    await Task.WhenAll(_controlNet.UnloadAsync(), _unet.UnloadAsync());

                // Decode Latents
                return await DecodeLatentsAsync(generateOptions, latents, cancellationToken);
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
            if (!options.Timesteps.IsNullOrEmpty())
                return options.Timesteps;

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
        protected override Task<DenseTensor<float>> PrepareLatentsAsync(GenerateOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps, CancellationToken cancellationToken = default)
        {
            return Task.FromResult(scheduler.CreateRandomSample(options.SchedulerOptions.GetScaledDimension(), scheduler.InitNoiseSigma));
        }


        /// <summary>
        /// Creates the Conditioning Scale tensor.
        /// </summary>
        /// <param name="conditioningScale">The conditioningScale.</param>
        /// <returns></returns>
        protected static DenseTensor<double> CreateConditioningScaleTensor(float conditioningScale)
        {
            return new DenseTensor<double>(new double[] { conditioningScale }, new int[] { 1 });
        }


        /// <summary>
        /// Prepares the control image.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns></returns>
        protected async Task<DenseTensor<float>> PrepareControlImage(GenerateOptions options)
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
