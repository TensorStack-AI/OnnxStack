using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.StableDiffusion.Schedulers.LatentConsistency;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers.LatentConsistency
{
    public abstract class LatentConsistencyDiffuser : DiffuserBase
    {

        /// <summary>
        /// Initializes a new instance of the <see cref="LatentConsistencyDiffuser"/> class.
        /// </summary>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public LatentConsistencyDiffuser(UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, MemoryModeType memoryMode, ILogger logger = default)
            : base(unet, vaeDecoder, vaeEncoder, memoryMode, logger) { }


        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override DiffuserPipelineType PipelineType => DiffuserPipelineType.LatentConsistency;


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
            // Get Scheduler
            using (var scheduler = GetScheduler(schedulerOptions))
            {
                // Get timesteps
                var timesteps = GetTimesteps(schedulerOptions, scheduler);

                // Create latent sample
                var latents = await PrepareLatentsAsync(promptOptions, schedulerOptions, scheduler, timesteps);

                // Get Guidance Scale Embedding
                var guidanceEmbeddings = GetGuidanceScaleEmbedding(schedulerOptions.GuidanceScale);

                // Denoised result
                DenseTensor<float> denoised = null;

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
                    var inputTensor = scheduler.ScaleInput(latents, timestep);
                    var timestepTensor = CreateTimestepTensor(timestep);

                    var outputChannels = 1;
                    var outputDimension = schedulerOptions.GetScaledDimension(outputChannels);
                    using (var inferenceParameters = new OnnxInferenceParameters(metadata))
                    {
                        inferenceParameters.AddInputTensor(inputTensor);
                        inferenceParameters.AddInputTensor(timestepTensor);
                        inferenceParameters.AddInputTensor(promptEmbeddings.PromptEmbeds);
                        if (inferenceParameters.InputCount == 4)
                            inferenceParameters.AddInputTensor(guidanceEmbeddings);
                        inferenceParameters.AddOutputBuffer(outputDimension);

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

                    ReportProgress(progressCallback, step, timesteps.Count, latents);
                    _logger?.LogEnd($"Step {step}/{timesteps.Count}", stepTime);
                }

                // Unload if required
                if (_memoryMode != MemoryModeType.Maximum)
                    await _unet.UnloadAsync();

                // Decode Latents
                return await DecodeLatentsAsync(promptOptions, schedulerOptions, denoised);
            }
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
                SchedulerType.LCM => new LCMScheduler(options),
                _ => default
            };
        }


        /// <summary>
        /// Gets the guidance scale embedding.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="embeddingDim">The embedding dim.</param>
        /// <returns></returns>
        protected DenseTensor<float> GetGuidanceScaleEmbedding(float guidance, int embeddingDim = 256)
        {
            var scale = (guidance - 1f) * 1000.0f;
            var halfDim = embeddingDim / 2;
            float log = MathF.Log(10000.0f) / (halfDim - 1);
            var emb = Enumerable.Range(0, halfDim)
                .Select(x => scale * MathF.Exp(-log * x))
                .ToArray();
            var embSin = emb.Select(MathF.Sin);
            var embCos = emb.Select(MathF.Cos);
            var guidanceEmbedding = embSin.Concat(embCos).ToArray();
            return new DenseTensor<float>(guidanceEmbedding, new[] { 1, embeddingDim });
        }
    }
}
