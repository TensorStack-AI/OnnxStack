using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Schedulers.StableDiffusion;
using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers.StableDiffusionXL
{
    public abstract class StableDiffusionXLDiffuser : DiffuserBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionXLDiffuser"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="onnxModelService">The onnx model service.</param>
        public StableDiffusionXLDiffuser(IOnnxModelService onnxModelService, IPromptService promptService, ILogger<StableDiffusionXLDiffuser> logger)
            : base(onnxModelService, promptService, logger) { }


        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override DiffuserPipelineType PipelineType => DiffuserPipelineType.StableDiffusionXL;


        /// <summary>
        /// Runs the scheduler steps.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <param name="performGuidance">if set to <c>true</c> [perform guidance].</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        protected override async Task<DenseTensor<float>> SchedulerStepAsync(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, PromptEmbeddingsResult promptEmbeddings, bool performGuidance, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default)
        {
            // Get Scheduler
            using (var scheduler = GetScheduler(schedulerOptions))
            {
                // Get timesteps
                var timesteps = GetTimesteps(schedulerOptions, scheduler);

                // Create latent sample
                var latents = await PrepareLatentsAsync(modelOptions, promptOptions, schedulerOptions, scheduler, timesteps);

                // Get Model metadata
                var metadata = _onnxModelService.GetModelMetadata(modelOptions, OnnxModelType.Unet);

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
                    var addTimeIds = GetAddTimeIds(schedulerOptions, performGuidance);

                    var outputChannels = performGuidance ? 2 : 1;
                    var outputDimension = schedulerOptions.GetScaledDimension(outputChannels);
                    using (var inferenceParameters = new OnnxInferenceParameters(metadata))
                    {
                        inferenceParameters.AddInputTensor(inputTensor);
                        inferenceParameters.AddInputTensor(timestepTensor);
                        inferenceParameters.AddInputTensor(promptEmbeddings.PromptEmbeds);
                        inferenceParameters.AddInputTensor(promptEmbeddings.PooledPromptEmbeds);
                        inferenceParameters.AddInputTensor(addTimeIds);
                        inferenceParameters.AddOutputBuffer(outputDimension);

                        var results = await _onnxModelService.RunInferenceAsync(modelOptions, OnnxModelType.Unet, inferenceParameters);
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

                    progressCallback?.Invoke(step, timesteps.Count);
                    _logger?.LogEnd($"Step {step}/{timesteps.Count}", stepTime);
                }

                // Decode Latents
                return await DecodeLatentsAsync(modelOptions, promptOptions, schedulerOptions, latents);
            }
        }


        /// <summary>
        /// Gets the add AddTimeIds.
        /// </summary>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <returns></returns>
        protected DenseTensor<float> GetAddTimeIds(SchedulerOptions schedulerOptions, bool performGuidance)
        {
            var addTimeIds = new float[]
            {
                schedulerOptions.Height, schedulerOptions.Width, //original_size
                0, 0, //crops_coords_top_left
                schedulerOptions.Height, schedulerOptions.Width //negative_target_size
            };
            var result = TensorHelper.CreateTensor(addTimeIds, new[] { 1, addTimeIds.Length });
            if (performGuidance)
                return result.Repeat(2);

            return result;
        }


        /// <summary>
        /// Gets the scheduler.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="schedulerConfig">The scheduler configuration.</param>
        /// <returns></returns>
        protected override IScheduler GetScheduler(SchedulerOptions options)
        {
            return options.SchedulerType switch
            {
                SchedulerType.LMS => new LMSScheduler(options),
                SchedulerType.Euler => new EulerScheduler(options),
                SchedulerType.EulerAncestral => new EulerAncestralScheduler(options),
                SchedulerType.DDPM => new DDPMScheduler(options),
                SchedulerType.DDIM => new DDIMScheduler(options),
                SchedulerType.KDPM2 => new KDPM2Scheduler(options),
                _ => default
            };
        }
    }
}
