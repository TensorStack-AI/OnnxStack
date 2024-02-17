using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Image;
using OnnxStack.Core.Video;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Pipelines
{
    public interface IPipeline
    {
        /// <summary>
        /// Gets the pipelines supported diffusers.
        /// </summary>
        IReadOnlyList<DiffuserType> SupportedDiffusers { get; }


        /// <summary>
        /// Gets the pipelines supported schedulers.
        /// </summary>
        IReadOnlyList<SchedulerType> SupportedSchedulers { get; }


        /// <summary>
        /// Gets the default scheduler options.
        /// </summary>
        SchedulerOptions DefaultSchedulerOptions { get; }


        /// <summary>
        /// Loads the pipeline.
        /// </summary>
        /// <returns></returns>
        Task LoadAsync();


        /// <summary>
        /// Unloads the pipeline.
        /// </summary>
        /// <returns></returns>
        Task UnloadAsync();


        /// <summary>
        /// Validates the inputs.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        void ValidateInputs(PromptOptions promptOptions, SchedulerOptions schedulerOptions);


        /// <summary>
        /// Runs the pipeline.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        Task<DenseTensor<float>> RunAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Runs the pipeline batch.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="batchOptions">The batch options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        IAsyncEnumerable<BatchResult> RunBatchAsync(BatchOptions batchOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Runs the pipeline returning the result as an OnnxImage.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        Task<OnnxImage> GenerateImageAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Runs the batch pipeline returning the result as an OnnxImage.
        /// </summary>
        /// <param name="batchOptions">The batch options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        IAsyncEnumerable<BatchImageResult> GenerateImageBatchAsync(BatchOptions batchOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Runs the pipeline returning the result as an OnnxVideo.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        Task<OnnxVideo> GenerateVideoAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Runs the batch pipeline returning the result as an OnnxVideo.
        /// </summary>
        /// <param name="batchOptions">The batch options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        IAsyncEnumerable<BatchVideoResult> GenerateVideoBatchAsync(BatchOptions batchOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);
    }
}