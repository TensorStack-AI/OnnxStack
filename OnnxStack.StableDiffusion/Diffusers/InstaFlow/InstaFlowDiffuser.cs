using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.StableDiffusion.Schedulers.InstaFlow;
using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers.InstaFlow
{
    public abstract class InstaFlowDiffuser : DiffuserBase
    {

        /// <summary>
        /// Initializes a new instance of the <see cref="InstaFlowDiffuser"/> class.
        /// </summary>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public InstaFlowDiffuser(UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, MemoryModeType memoryMode, ILogger logger = default)
            : base(unet, vaeDecoder, vaeEncoder, memoryMode, logger) { }

        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override DiffuserPipelineType PipelineType => DiffuserPipelineType.InstaFlow;


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

                // Get Model metadata
                var metadata = await _unet.GetMetadataAsync();

                // Get the distilled Timestep
                var distilledTimestep = 1.0f / timesteps.Count;

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

                    var outputChannels = performGuidance ? 2 : 1;
                    var outputDimension = schedulerOptions.GetScaledDimension(outputChannels);
                    using (var inferenceParameters = new OnnxInferenceParameters(metadata))
                    {
                        inferenceParameters.AddInputTensor(inputTensor);
                        inferenceParameters.AddInputTensor(timestepTensor);
                        inferenceParameters.AddInputTensor(promptEmbeddings.PromptEmbeds);
                        inferenceParameters.AddOutputBuffer(outputDimension);

                        var results = await _unet.RunInferenceAsync(inferenceParameters);
                        using (var result = results.First())
                        {
                            var noisePred = result.ToDenseTensor();

                            // Perform guidance
                            if (performGuidance)
                                noisePred = PerformGuidance(noisePred, schedulerOptions.GuidanceScale);

                            // Scheduler Step
                            latents = scheduler.Step(noisePred, timestep, latents).Result;

                            latents = noisePred
                                .MultiplyTensorByFloat(distilledTimestep)
                                .AddTensors(latents);
                        }
                    }

                    ReportProgress(progressCallback, step, timesteps.Count, latents);
                    _logger?.LogEnd($"Step {step}/{timesteps.Count}", stepTime);
                }

                // Unload if required
                if (_memoryMode != MemoryModeType.Maximum)
                    await _unet.UnloadAsync();

                // Decode Latents
                return await DecodeLatentsAsync(promptOptions, schedulerOptions, latents);
            }
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
            timestepTensor.Fill(timestep);
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
                SchedulerType.InstaFlow => new InstaFlowScheduler(options),
                _ => default
            };
        }
    }
}
