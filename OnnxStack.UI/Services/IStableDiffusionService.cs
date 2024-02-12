using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Image;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.UI.Services
{
    public interface IStableDiffusionService
    {
        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <returns></returns>
        Task<bool> LoadModelAsync(StableDiffusionModelSet model);


        /// <summary>
        /// Unloads the model.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <returns></returns>
        Task<bool> UnloadModelAsync(StableDiffusionModelSet model);

        /// <summary>
        /// Determines whether the specified model is loaded
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <returns>
        ///   <c>true</c> if the specified model is loaded; otherwise, <c>false</c>.
        /// </returns>
        bool IsModelLoaded(StableDiffusionModelSet model);


        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <returns></returns>
        Task<bool> LoadControlNetModelAsync(ControlNetModelSet model);


        /// <summary>
        /// Unloads the model.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <returns></returns>
        Task<bool> UnloadControlNetModelAsync(ControlNetModelSet model);

        /// <summary>
        /// Determines whether the specified model is loaded
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <returns>
        ///   <c>true</c> if the specified model is loaded; otherwise, <c>false</c>.
        /// </returns>
        bool IsControlNetModelLoaded(ControlNetModelSet model);

        /// <summary>
        /// Generates the StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The Scheduler options.</param>
        /// <param name="progressCallback">The callback used to provide progess of the current InferenceSteps.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>The diffusion result as <see cref="DenseTensor<float>"/></returns>
        Task<OnnxImage> GenerateAsync(ModelOptions model, PromptOptions prompt, SchedulerOptions options, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);

        /// <summary>
        /// Generates a batch of StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="batchOptions">The batch options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        IAsyncEnumerable<BatchResult> GenerateBatchAsync(ModelOptions model, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);
    }
}