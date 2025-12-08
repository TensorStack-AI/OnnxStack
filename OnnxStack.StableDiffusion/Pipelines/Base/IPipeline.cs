using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Image;
using OnnxStack.Core.Video;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Pipelines
{
    public interface IPipeline
    {

        /// <summary>
        /// Gets the name.
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        PipelineType PipelineType { get; }

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
        /// Unloads the pipeline.
        /// </summary>
        /// <returns></returns>
        Task UnloadAsync();


        /// <summary>
        /// Runs the pipeline.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        Task<DenseTensor<float>> RunAsync(GenerateOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Runs the pipeline batch.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        IAsyncEnumerable<BatchResult> RunBatchAsync(GenerateBatchOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Runs the pipeline returning the result as an OnnxImage.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        Task<OnnxImage> GenerateAsync(GenerateOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Runs the batch pipeline returning the result as an OnnxImage.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        IAsyncEnumerable<BatchImageResult> GenerateBatchAsync(GenerateBatchOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Runs the pipeline returning the result as an OnnxVideo.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        Task<OnnxVideo> GenerateVideoAsync(GenerateOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Runs the pipeline returning the result as an OnnxVideo.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        IAsyncEnumerable<OnnxImage> GenerateVideoFramesAsync(GenerateOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Runs the video stream pipeline returning each frame as an OnnxImage.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="videoFrames">The video frames.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        IAsyncEnumerable<OnnxImage> GenerateVideoStreamAsync(GenerateOptions options, IAsyncEnumerable<OnnxImage> videoFrames, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);
    }
}