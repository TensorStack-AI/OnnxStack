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
        public StableCascadeDiffuser(UNetConditionModel priorUnet, UNetConditionModel decoderUnet, AutoEncoderModel decoderVqgan, AutoEncoderModel imageEncoder, MemoryModeType memoryMode, ILogger logger = default)
            : base(priorUnet, decoderVqgan, imageEncoder, memoryMode, logger)
        {
            _decoderUnet = decoderUnet;
            _latentDimScale = 10.67f;
            _resolutionMultiple = 42.67f;
            _clipImageChannels = 768;
        }

        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override DiffuserPipelineType PipelineType => DiffuserPipelineType.StableCascade;


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
        /// Runs the scheduler steps.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <param name="performGuidance">if set to <c>true</c> [perform guidance].</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async Task<DenseTensor<float>> DiffuseAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions, PromptEmbeddingsResult promptEmbeddings, bool performGuidance, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var decodeSchedulerOptions = schedulerOptions with
            {
                InferenceSteps = schedulerOptions.InferenceSteps2,
                GuidanceScale = schedulerOptions.GuidanceScale2
            };

            var priorPromptEmbeddings = promptEmbeddings;
            var decoderPromptEmbeddings = promptEmbeddings;
            var priorPerformGuidance = schedulerOptions.GuidanceScale > 0;
            var decoderPerformGuidance = decodeSchedulerOptions.GuidanceScale > 0;
            if (performGuidance)
            {
                if (!priorPerformGuidance)
                    priorPromptEmbeddings = SplitPromptEmbeddings(promptEmbeddings);
                if (!decoderPerformGuidance)
                    decoderPromptEmbeddings = SplitPromptEmbeddings(promptEmbeddings);
            }

            // Prior Unet
            var priorLatents = await DiffusePriorAsync(promptOptions, schedulerOptions, priorPromptEmbeddings, priorPerformGuidance, progressCallback, cancellationToken);

            // Decoder Unet
            var decoderLatents = await DiffuseDecodeAsync(promptOptions, priorLatents, decodeSchedulerOptions, decoderPromptEmbeddings, decoderPerformGuidance, progressCallback, cancellationToken);

            // Decode Latents
            return await DecodeLatentsAsync(promptOptions, schedulerOptions, decoderLatents);
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
        protected async Task<DenseTensor<float>> DiffusePriorAsync(PromptOptions prompt, SchedulerOptions schedulerOptions, PromptEmbeddingsResult promptEmbeddings, bool performGuidance, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            using (var scheduler = GetScheduler(schedulerOptions))
            {
                // Get timesteps
                var timesteps = GetTimesteps(schedulerOptions, scheduler);

                // Create latent sample
                var latents = await PrepareLatentsAsync(prompt, schedulerOptions, scheduler, timesteps);

                var encodedImage = await EncodeImageAsync(prompt, performGuidance);

                // Get Model metadata
                var metadata = await _unet.GetMetadataAsync();

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
                    var timestepTensor = CreateTimestepTensor(inputLatent, timestep);
                    using (var inferenceParameters = new OnnxInferenceParameters(metadata))
                    {
                        inferenceParameters.AddInputTensor(inputTensor);
                        inferenceParameters.AddInputTensor(timestepTensor);
                        inferenceParameters.AddInputTensor(promptEmbeddings.PooledPromptEmbeds);
                        inferenceParameters.AddInputTensor(promptEmbeddings.PromptEmbeds);
                        inferenceParameters.AddInputTensor(encodedImage);
                        inferenceParameters.AddOutputBuffer(inputTensor.Dimensions);

                        var results = await _unet.RunInferenceAsync(inferenceParameters);
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

                    ReportProgress(progressCallback, step, timesteps.Count, latents);
                    _logger?.LogEnd(LogLevel.Debug, $"Prior Step {step}/{timesteps.Count}", stepTime);
                }

                // Unload if required
                if (_memoryMode == MemoryModeType.Minimum)
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
        protected async Task<DenseTensor<float>> DiffuseDecodeAsync(PromptOptions prompt, DenseTensor<float> priorLatents, SchedulerOptions schedulerOptions, PromptEmbeddingsResult promptEmbeddings, bool performGuidance, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            using (var scheduler = GetScheduler(schedulerOptions))
            {
                // Get timesteps
                var timesteps = GetTimesteps(schedulerOptions, scheduler);

                // Create latent sample
                var latents = await PrepareDecoderLatentsAsync(prompt, schedulerOptions, scheduler, timesteps, priorLatents);

                // Get Model metadata
                var metadata = await _decoderUnet.GetMetadataAsync();

                var effnet = !performGuidance
                    ? priorLatents
                    : priorLatents.Repeat(2);

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
                    var timestepTensor = CreateTimestepTensor(inputLatent, timestep);
                    using (var inferenceParameters = new OnnxInferenceParameters(metadata))
                    {
                        inferenceParameters.AddInputTensor(inputTensor);
                        inferenceParameters.AddInputTensor(timestepTensor);
                        inferenceParameters.AddInputTensor(promptEmbeddings.PooledPromptEmbeds);
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

                    ReportProgress(progressCallback, step, timesteps.Count, latents);
                    _logger?.LogEnd(LogLevel.Debug, $"Decoder Step {step}/{timesteps.Count}", stepTime);
                }

                // Unload if required
                if (_memoryMode == MemoryModeType.Minimum)
                    await _unet.UnloadAsync();

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
        protected override async Task<DenseTensor<float>> DecodeLatentsAsync(PromptOptions prompt, SchedulerOptions options, DenseTensor<float> latents)
        {
            latents = latents.MultiplyBy(_vaeDecoder.ScaleFactor);

            var outputDim = new[] { 1, 3, options.Height, options.Width };
            var metadata = await _vaeDecoder.GetMetadataAsync();
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(latents);
                inferenceParameters.AddOutputBuffer(outputDim);

                var results = await _vaeDecoder.RunInferenceAsync(inferenceParameters);
                using (var imageResult = results.First())
                {
                    // Unload if required
                    if (_memoryMode == MemoryModeType.Minimum)
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
        protected override Task<DenseTensor<float>> PrepareLatentsAsync(PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps)
        {
            var latents = scheduler.CreateRandomSample(new[]
            {
               1, 16,
               (int)Math.Ceiling(options.Height / ResolutionMultiple),
               (int)Math.Ceiling(options.Width / ResolutionMultiple)
           }, scheduler.InitNoiseSigma);
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
        protected virtual Task<DenseTensor<float>> PrepareDecoderLatentsAsync(PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps, DenseTensor<float> priorLatents)
        {
            var latents = scheduler.CreateRandomSample(new[]
            {
                1, 4,
                (int)(priorLatents.Dimensions[2] * LatentDimScale),
                (int)(priorLatents.Dimensions[3] * LatentDimScale)
            }, scheduler.InitNoiseSigma);
            return Task.FromResult(latents);
        }


        /// <summary>
        /// Encodes the image.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="performGuidance">if set to <c>true</c> [perform guidance].</param>
        /// <returns></returns>
        protected virtual Task<DenseTensor<float>> EncodeImageAsync(PromptOptions prompt, bool performGuidance)
        {
            return Task.FromResult(new DenseTensor<float>(new[] { performGuidance ? 2 : 1, 1, _clipImageChannels }));
        }


        /// <summary>
        /// Creates the timestep tensor.
        /// </summary>
        /// <param name="latents">The latents.</param>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        private DenseTensor<float> CreateTimestepTensor(DenseTensor<float> latents, int timestep)
        {
            var timestepTensor = new DenseTensor<float>(new[] { latents.Dimensions[0] });
            timestepTensor.Fill(timestep / 1000f);
            return timestepTensor;
        }


        /// <summary>
        /// Splits the prompt embeddings, Removes unconditional embeddings
        /// </summary>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <returns></returns>
        private PromptEmbeddingsResult SplitPromptEmbeddings(PromptEmbeddingsResult promptEmbeddings)
        {
            return promptEmbeddings.PooledPromptEmbeds is null
                     ? new PromptEmbeddingsResult(promptEmbeddings.PromptEmbeds.SplitBatch().Last())
                     : new PromptEmbeddingsResult(promptEmbeddings.PromptEmbeds.SplitBatch().Last(), promptEmbeddings.PooledPromptEmbeds.SplitBatch().Last());
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
                SchedulerType.DDPM => new DDPMScheduler(options),
                SchedulerType.DDPMWuerstchen => new DDPMWuerstchenScheduler(options),
                _ => default
            };
        }
    }
}
