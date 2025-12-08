using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.StableDiffusion.Schedulers.StableDiffusion;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers.StableCascade
{
    public abstract class StableCascadeDiffuser : DiffuserBase
    {
        private readonly float _latentDimScale;
        private readonly float _resolutionMultiple;
        private readonly int _clipImageChannels;
        private readonly UNetConditionModel _decoderUnet;

        /// <summary>
        /// Initializes a new instance of the <see cref="StableCascadeDiffuser"/> class.
        /// </summary>
        /// <param name="priorUnet">The prior unet.</param>
        /// <param name="decoderUnet">The decoder unet.</param>
        /// <param name="decoderVqgan">The decoder vqgan.</param>
        /// <param name="imageEncoder">The image encoder.</param>
        /// <param name="memoryMode">The memory mode.</param>
        /// <param name="logger">The logger.</param>
        public StableCascadeDiffuser(UNetConditionModel priorUnet, UNetConditionModel decoderUnet, AutoEncoderModel decoderVqgan, AutoEncoderModel imageEncoder, ILogger logger = default)
            : base(priorUnet, decoderVqgan, imageEncoder, logger)
        {
            _decoderUnet = decoderUnet;
            _latentDimScale = 10.67f;
            _resolutionMultiple = 42.67f;
            _clipImageChannels = 768;
        }

        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override PipelineType PipelineType => PipelineType.StableCascade;


        /// <summary>
        /// Multiplier to determine the VQ latent space size from the image embeddings. If the image embeddings are
        /// height=24 and width = 24, the VQ latent shape needs to be height=int (24*10.67)=256 and
        /// width = int(24 * 10.67) = 256 in order to match the training conditions.
        /// </summary>
        protected float LatentDimScale => _latentDimScale;


        /// <summary>
        /// Default resolution for multiple images generated
        /// </summary>
        protected float ResolutionMultiple => _resolutionMultiple;


        /// <summary>
        /// Gets the clip image channels.
        /// </summary>
        protected int ClipImageChannels => _clipImageChannels;


        /// <summary>
        /// Check if we should run guidance.
        /// </summary>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <returns></returns>
        protected override bool ShouldPerformGuidance(SchedulerOptions schedulerOptions)
        {
            return false;
        }


        /// <summary>
        /// Runs the scheduler steps.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async Task<DenseTensor<float>> DiffuseAsync(DiffuseOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var generateOptions = options.GenerateOptions;
            var schedulerOptions = generateOptions.SchedulerOptions;


            // Prior Unet
            var priorPerformGuidance = schedulerOptions.GuidanceScale > 0;
            var priorPromptEmbeds = options.PromptEmbeddings.GetPromptEmbeds(priorPerformGuidance);
            var priorPooledPromptEmbeds = options.PromptEmbeddings.GetPooledPromptEmbeds(priorPerformGuidance);
            var priorLatents = await DiffusePriorAsync(generateOptions, priorPromptEmbeds, priorPooledPromptEmbeds, priorPerformGuidance, progressCallback, cancellationToken);


            // Decoder Unet
            var decodeSchedulerOptions = schedulerOptions with
            {
                InferenceSteps = schedulerOptions.InferenceSteps2,
                GuidanceScale = schedulerOptions.GuidanceScale2
            };
            var decoderPerformGuidance = decodeSchedulerOptions.GuidanceScale > 0;
            var decoderPromptEmbeds = options.PromptEmbeddings.GetPromptEmbeds(decoderPerformGuidance);
            var decoderPooledPromptEmbeds = options.PromptEmbeddings.GetPooledPromptEmbeds(decoderPerformGuidance);
            var decoderLatents = await DiffuseDecodeAsync(generateOptions, decodeSchedulerOptions, priorLatents, decoderPromptEmbeds, decoderPooledPromptEmbeds, decoderPerformGuidance, progressCallback, cancellationToken);


            // Decode Latents
            return await DecodeLatentsAsync(generateOptions, decoderLatents, cancellationToken);
        }



        /// <summary>
        /// Run the Prior UNET diffusion
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <param name="performGuidance">if set to <c>true</c> [perform guidance].</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        protected async Task<DenseTensor<float>> DiffusePriorAsync(GenerateOptions options, DenseTensor<float> promptEmbeds, DenseTensor<float> pooledPromptEmbeds, bool performGuidance, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            using (var scheduler = GetScheduler(options.SchedulerOptions))
            {
                // Get Model metadata
                var metadata = await _unet.LoadAsync(cancellationToken: cancellationToken);

                // Get timesteps
                var timesteps = GetTimesteps(options.SchedulerOptions, scheduler);

                // Create latent sample
                progressCallback.Notify("Prepare Input...");
                var latents = await PrepareLatentsAsync(options, scheduler, timesteps, cancellationToken);

                var encodedImage = await EncodeImageAsync(options, performGuidance, cancellationToken);

                // Loop though the timesteps
                var step = 0;
                ReportProgress(progressCallback, "Step", 0, timesteps.Count, 0);
                foreach (var timestep in timesteps)
                {
                    step++;
                    var stepTime = Stopwatch.GetTimestamp();
                    cancellationToken.ThrowIfCancellationRequested();

                    // Create input tensor.
                    var inputLatent = performGuidance ? latents.Repeat(2) : latents;
                    var inputTensor = scheduler.ScaleInput(inputLatent, timestep);
                    var timestepTensor = CreateTimestepTensor(inputLatent, timestep);
                    using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
                    {
                        inferenceParameters.AddInputTensor(inputTensor);
                        inferenceParameters.AddInputTensor(timestepTensor);
                        inferenceParameters.AddInputTensor(pooledPromptEmbeds);
                        inferenceParameters.AddInputTensor(promptEmbeds);
                        inferenceParameters.AddInputTensor(encodedImage);
                        inferenceParameters.AddOutputBuffer(inputTensor.Dimensions);

                        var results = await _unet.RunInferenceAsync(inferenceParameters);
                        using (var result = results.First())
                        {
                            var noisePred = result.ToDenseTensor();

                            // Perform guidance
                            if (performGuidance)
                                noisePred = PerformGuidance(noisePred, options.SchedulerOptions.GuidanceScale);

                            // Scheduler Step
                            latents = scheduler.Step(noisePred, timestep, latents).Result;
                        }
                    }

                    ReportProgress(progressCallback, "Step", step, timesteps.Count, stepTime, latents);
                    _logger?.LogEnd(LogLevel.Debug, $"Prior Step {step}/{timesteps.Count}", stepTime);
                }

                // Unload if required
                if (options.IsLowMemoryComputeEnabled)
                    await _unet.UnloadAsync();

                return latents;
            }
        }




        /// <summary>
        /// Run the Decoder UNET diffusion
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="priorLatents">The prior latents.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <param name="performGuidance">if set to <c>true</c> [perform guidance].</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        protected async Task<DenseTensor<float>> DiffuseDecodeAsync(GenerateOptions options, SchedulerOptions schedulerOptions, DenseTensor<float> priorLatents, DenseTensor<float> promptEmbeds, DenseTensor<float> pooledPromptEmbeds, bool performGuidance, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            using (var scheduler = GetScheduler(schedulerOptions))
            {
                // Get Model metadata
                var metadata = await _decoderUnet.LoadAsync(cancellationToken: cancellationToken);

                // Get timesteps
                var timesteps = GetTimesteps(schedulerOptions, scheduler);

                // Create latent sample
                progressCallback.Notify("Prepare Input...");
                var latents = await PrepareDecoderLatentsAsync(options, scheduler, timesteps, priorLatents, cancellationToken);

                var effnet = !performGuidance
                    ? priorLatents
                    : priorLatents.Repeat(2);

                // Loop though the timesteps
                var step = 0;
                ReportProgress(progressCallback, "Step", 0, timesteps.Count, 0);
                foreach (var timestep in timesteps)
                {
                    step++;
                    var stepTime = Stopwatch.GetTimestamp();
                    cancellationToken.ThrowIfCancellationRequested();

                    // Create input tensor.
                    var inputLatent = performGuidance ? latents.Repeat(2) : latents;
                    var inputTensor = scheduler.ScaleInput(inputLatent, timestep);
                    var timestepTensor = CreateTimestepTensor(inputLatent, timestep);
                    using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
                    {
                        inferenceParameters.AddInputTensor(inputTensor);
                        inferenceParameters.AddInputTensor(timestepTensor);
                        inferenceParameters.AddInputTensor(pooledPromptEmbeds);
                        inferenceParameters.AddInputTensor(effnet);
                        inferenceParameters.AddOutputBuffer(inputTensor.Dimensions);

                        var results = await _decoderUnet.RunInferenceAsync(inferenceParameters);
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

                    ReportProgress(progressCallback, "Step", step, timesteps.Count, stepTime, latents);
                    _logger?.LogEnd(LogLevel.Debug, $"Decoder Step {step}/{timesteps.Count}", stepTime);
                }

                // Unload if required
                if (options.IsLowMemoryComputeEnabled)
                    await _decoderUnet.UnloadAsync();

                return latents;
            }
        }


        /// <summary>
        /// Decodes the latents.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="latents">The latents.</param>
        /// <returns></returns>
        protected override async Task<DenseTensor<float>> DecodeLatentsAsync(GenerateOptions options, DenseTensor<float> latents, CancellationToken cancellationToken = default)
        {
            latents = latents.MultiplyBy(_vaeDecoder.ScaleFactor);

            var outputDim = new[] { 1, 3, options.SchedulerOptions.Height, options.SchedulerOptions.Width };
            var metadata = await _vaeDecoder.LoadAsync(cancellationToken: cancellationToken);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
            {
                inferenceParameters.AddInputTensor(latents);
                inferenceParameters.AddOutputBuffer(outputDim);

                var results = await _vaeDecoder.RunInferenceAsync(inferenceParameters);
                using (var imageResult = results.First())
                {
                    // Unload if required
                    if (options.IsLowMemoryDecoderEnabled)
                        await _vaeDecoder.UnloadAsync();

                    return imageResult
                        .ToArray()
                        .AsSpan()
                        .NormalizeOneToOne()
                        .ToDenseTensor(outputDim);
                }
            }
        }


        /// <summary>
        /// Prepares the input latents.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <returns></returns>
        protected override Task<DenseTensor<float>> PrepareLatentsAsync(GenerateOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps, CancellationToken cancellationToken = default)
        {
            var latents = scheduler.CreateRandomSample(
            [
               1, 16,
               (int)Math.Ceiling(options.SchedulerOptions.Height / ResolutionMultiple),
               (int)Math.Ceiling(options.SchedulerOptions.Width / ResolutionMultiple)
            ], scheduler.InitNoiseSigma);
            return Task.FromResult(latents);
        }


        /// <summary>
        /// Prepares the decoder latents.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <param name="priorLatents">The prior latents.</param>
        /// <returns></returns>
        protected virtual Task<DenseTensor<float>> PrepareDecoderLatentsAsync(GenerateOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps, DenseTensor<float> priorLatents, CancellationToken cancellationToken = default)
        {
            var latents = scheduler.CreateRandomSample(
            [
                1, 4,
                (int)(priorLatents.Dimensions[2] * LatentDimScale),
                (int)(priorLatents.Dimensions[3] * LatentDimScale)
            ], scheduler.InitNoiseSigma);
            return Task.FromResult(latents);
        }


        /// <summary>
        /// Encodes the image.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="performGuidance">if set to <c>true</c> [perform guidance].</param>
        /// <returns></returns>
        protected virtual Task<DenseTensor<float>> EncodeImageAsync(GenerateOptions options, bool performGuidance, CancellationToken cancellationToken = default)
        {
            return Task.FromResult(new DenseTensor<float>([performGuidance ? 2 : 1, 1, _clipImageChannels]));
        }


        /// <summary>
        /// Creates the timestep tensor.
        /// </summary>
        /// <param name="latents">The latents.</param>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        private DenseTensor<float> CreateTimestepTensor(DenseTensor<float> latents, int timestep)
        {
            var timestepTensor = new DenseTensor<float>([latents.Dimensions[0]]);
            timestepTensor.Fill(timestep / 1000f);
            return timestepTensor;
        }


        /// <summary>
        /// Gets the scheduler.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns></returns>
        protected override IScheduler GetScheduler(SchedulerOptions options)
        {
            return options.SchedulerType switch
            {
                SchedulerType.DDPMWuerstchen => new DDPMWuerstchenScheduler(options),
                _ => default
            };
        }
    }
}
