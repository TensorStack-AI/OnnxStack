using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers
{
    public interface IDiffuser
    {

        /// <summary>
        /// Gets the type of the diffuser.
        /// </summary>
        DiffuserType DiffuserType { get; }


        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        DiffuserPipelineType PipelineType { get; }


        /// <summary>
        /// Runs the stable diffusion loop
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        Task<DenseTensor<float>> DiffuseAsync(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default);


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
        IAsyncEnumerable<DenseTensor<float>> DiffuseBatchAsync(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<int, int, int, int> progressCallback = null, CancellationToken cancellationToken = default);
    }
}
