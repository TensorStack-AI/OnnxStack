using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers.StableDiffusionXL;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Threading;
using System;
using OnnxStack.StableDiffusion.Schedulers.LatentConsistency;

namespace OnnxStack.StableDiffusion.Diffusers.LatentConsistencyXL
{
    public abstract class LatentConsistencyXLDiffuser : StableDiffusionXLDiffuser
    {
        protected LatentConsistencyXLDiffuser(IOnnxModelService onnxModelService, IPromptService promptService, ILogger<StableDiffusionXLDiffuser> logger)
            : base(onnxModelService, promptService, logger) { }


        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override DiffuserPipelineType PipelineType => DiffuserPipelineType.LatentConsistencyXL;


        /// <summary>
        /// Runs the stable diffusion loop
        /// </summary>
        /// <param name="modelOptions"></param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="progressCallback"></param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override Task<DenseTensor<float>> DiffuseAsync(ModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            // LCM does not support negative prompting
            promptOptions.NegativePrompt = string.Empty;
            return base.DiffuseAsync(modelOptions, promptOptions, schedulerOptions, progressCallback, cancellationToken);
        }


        /// <summary>
        /// Runs the stable diffusion batch loop
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="batchOptions">The batch options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override IAsyncEnumerable<BatchResult> DiffuseBatchAsync(ModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            // LCM does not support negative prompting
            promptOptions.NegativePrompt = string.Empty;
            return base.DiffuseBatchAsync(modelOptions, promptOptions, schedulerOptions, batchOptions, progressCallback, cancellationToken);
        }


        /// <summary>
        /// Gets the scheduler.
        /// </summary>
        /// <param name="prompt"></param>
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
    }
}
