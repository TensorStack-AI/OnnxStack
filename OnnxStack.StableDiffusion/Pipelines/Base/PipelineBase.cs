using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Image;
using OnnxStack.Core.Video;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Pipelines
{
    public abstract class PipelineBase : IPipeline
    {
        protected readonly string _name;
        protected readonly ILogger _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="PipelineBase"/> class.
        /// </summary>
        /// <param name="name">The pipeline name.</param>
        /// <param name="logger">The logger.</param>
        protected PipelineBase(string name, ILogger logger)
        {
            _name = name;
            _logger = logger;
        }


        /// <summary>
        /// Gets the pipelines friendly name.
        /// </summary>
        public string Name => _name;


        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public abstract PipelineType PipelineType { get; }


        /// <summary>
        /// Gets the pipelines supported diffusers.
        /// </summary>
        public abstract IReadOnlyList<DiffuserType> SupportedDiffusers { get; }


        /// <summary>
        /// Gets the pipelines supported schedulers.
        /// </summary>
        public abstract IReadOnlyList<SchedulerType> SupportedSchedulers { get; }


        /// <summary>
        /// Gets the default scheduler options.
        /// </summary>
        public abstract SchedulerOptions DefaultSchedulerOptions { get; }


        /// <summary>
        /// Unloads the pipeline.
        /// </summary>
        /// <returns></returns>
        public abstract Task UnloadAsync();


        /// <summary>
        /// Runs the pipeline.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public abstract Task<DenseTensor<float>> RunAsync(GenerateOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Runs the pipeline batch.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public abstract IAsyncEnumerable<BatchResult> RunBatchAsync(GenerateBatchOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Runs the pipeline returning the result as an OnnxImage.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public abstract Task<OnnxImage> GenerateAsync(GenerateOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Runs the batch pipeline returning the result as an OnnxImage.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public abstract IAsyncEnumerable<BatchImageResult> GenerateBatchAsync(GenerateBatchOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Runs the pipeline returning the result as an OnnxVideo.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public abstract Task<OnnxVideo> GenerateVideoAsync(GenerateOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Runs the pipeline returning each frame an OnnxImage.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public abstract IAsyncEnumerable<OnnxImage> GenerateVideoFramesAsync(GenerateOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Runs the video stream pipeline returning each frame as an OnnxImage.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="videoFrames">The video frames.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public abstract IAsyncEnumerable<OnnxImage> GenerateVideoStreamAsync(GenerateOptions options, IAsyncEnumerable<OnnxImage> videoFrames, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Creates the diffuser.
        /// </summary>
        /// <param name="diffuserType">Type of the diffuser.</param>
        /// <param name="controlNetModel">The control net model.</param>
        /// <returns></returns>
        protected abstract IDiffuser CreateDiffuser(DiffuserType diffuserType, ControlNetModel controlNetModel);


        /// <summary>
        /// Runs the Diffusion process and returns an image.
        /// </summary>
        /// <param name="diffuser">The diffuser.</param>
        /// <param name="options">The options.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        protected async Task<DenseTensor<float>> DiffuseImageAsync(IDiffuser diffuser, GenerateOptions options, PromptEmbeddingsResult promptEmbeddings, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var diffuseTime = Stopwatch.GetTimestamp();
            _logger?.LogInformation("Image Diffuser starting...");

            var imageOptions = new DiffuseOptions(options, promptEmbeddings);
            var schedulerResult = await diffuser.DiffuseAsync(imageOptions, progressCallback, cancellationToken);

            progressCallback?.Report(new DiffusionProgress(diffuseTime));
            _logger?.LogEnd($"Image Diffuser complete", diffuseTime);
            return schedulerResult;
        }


        /// <summary>
        /// Runs the Diffusion process and returns a video.
        /// </summary>
        /// <param name="diffuser">The diffuser.</param>
        /// <param name="options">The options.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        protected async IAsyncEnumerable<DenseTensor<float>> DiffuseVideoAsync(IDiffuser diffuser, GenerateOptions options, PromptEmbeddingsResult promptEmbeddings, IProgress<DiffusionProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var diffuseTime = Stopwatch.GetTimestamp();
            _logger?.LogInformation("Video Diffuser starting...");

            var videoOptions = new DiffuseOptions(options, promptEmbeddings);
            await foreach (var frameResultTensor in diffuser.DiffuseVideoAsync(videoOptions, progressCallback, cancellationToken))
            {
                yield return frameResultTensor;
            }

            progressCallback?.Report(new DiffusionProgress(diffuseTime));
            _logger?.LogEnd($"Video Diffuser complete", diffuseTime);
        }


        /// <summary>
        /// Reports the progress.
        /// </summary>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="progress">The progress.</param>
        /// <param name="progressMax">The progress maximum.</param>
        /// <param name="subProgress">The sub progress.</param>
        /// <param name="subProgressMax">The sub progress maximum.</param>
        /// <param name="output">The output.</param>
        protected void ReportBatchProgress(IProgress<DiffusionProgress> progressCallback, int progress, int progressMax, DenseTensor<float> progressTensor, long elapsed)
        {
            progressCallback?.Report(new DiffusionProgress(elapsed)
            {
                BatchMax = progressMax,
                BatchValue = progress,
                BatchTensor = progressTensor
            });
        }


        /// <summary>
        /// Creates the batch callback.
        /// </summary>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="batchCount">The batch count.</param>
        /// <param name="batchIndex">Index of the batch.</param>
        /// <returns></returns>
        protected IProgress<DiffusionProgress> CreateBatchCallback(IProgress<DiffusionProgress> progressCallback, int batchCount, Func<int> batchIndex)
        {
            if (progressCallback == null)
                return progressCallback;

            return new Progress<DiffusionProgress>((DiffusionProgress progress) => progressCallback?.Report(new DiffusionProgress
            {
                StepMax = progress.StepMax,
                StepValue = progress.StepValue,
                StepTensor = progress.StepTensor,
                BatchMax = batchCount,
                BatchValue = batchIndex(),
                BatchTensor = progress.BatchTensor,
                Message = progress.Message,
                Elapsed = progress.Elapsed
            }));
        }


        /// <summary>
        /// Logs the StableDiffusionModelSet information.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="logger">The logger.</param>
        protected static void LogPipelineInfo(StableDiffusionModelSet modelSet, ILogger logger)
        {
            if (logger is null)
                return;

            logger?.LogInformation("[{PipelineType}] - {Name}", modelSet.PipelineType.ToString(), modelSet.Name);
            if (modelSet.TextEncoderConfig != null)
                logger?.LogInformation("[{PipelineType}] - TextEncoder - Provider: {Name}, Path: {OnnxModelPath}", modelSet.PipelineType.ToString(), modelSet.TextEncoderConfig.ExecutionProvider.Name, modelSet.TextEncoderConfig.OnnxModelPath);
            if (modelSet.TextEncoder2Config != null)
                logger?.LogInformation("[{PipelineType}] - TextEncoder2 - Provider: {Name}, Path: {OnnxModelPath}", modelSet.PipelineType.ToString(), modelSet.TextEncoder2Config.ExecutionProvider.Name, modelSet.TextEncoder2Config.OnnxModelPath);
            if (modelSet.TextEncoder3Config != null)
                logger?.LogInformation("[{PipelineType}] - TextEncoder3 - Provider: {Name}, Path: {OnnxModelPath}", modelSet.PipelineType.ToString(), modelSet.TextEncoder3Config.ExecutionProvider.Name, modelSet.TextEncoder3Config.OnnxModelPath);
            if (modelSet.VaeEncoderConfig != null)
                logger?.LogInformation("[{PipelineType}] - VaeEncoder - Provider: {Name}, Path: {OnnxModelPath}", modelSet.PipelineType.ToString(), modelSet.VaeEncoderConfig.ExecutionProvider.Name, modelSet.VaeEncoderConfig.OnnxModelPath);
            if (modelSet.VaeDecoderConfig != null)
                logger?.LogInformation("[{PipelineType}] - VaeDecoder - Provider: {Name}, Path: {OnnxModelPath}", modelSet.PipelineType.ToString(), modelSet.VaeDecoderConfig.ExecutionProvider.Name, modelSet.VaeDecoderConfig.OnnxModelPath);
            if (modelSet.UnetConfig != null)
                logger?.LogInformation("[{PipelineType}] - Unet - Provider: {Name}, Path: {OnnxModelPath}", modelSet.PipelineType.ToString(), modelSet.UnetConfig.ExecutionProvider.Name, modelSet.UnetConfig.OnnxModelPath);
            if (modelSet.Unet2Config != null)
                logger?.LogInformation("[{PipelineType}] - Unet2 - Provider: {Name}, Path: {OnnxModelPath}", modelSet.PipelineType.ToString(), modelSet.Unet2Config.ExecutionProvider.Name, modelSet.Unet2Config.OnnxModelPath);
            if (modelSet.FlowEstimationConfig != null)
                logger?.LogInformation("[{PipelineType}] - FlowEstimation - Provider: {Name}, Path: {OnnxModelPath}", modelSet.PipelineType.ToString(), modelSet.FlowEstimationConfig.ExecutionProvider.Name, modelSet.FlowEstimationConfig.OnnxModelPath);
            if (modelSet.ResampleModelConfig != null)
                logger?.LogInformation("[{PipelineType}] - Resample - Provider: {Name}, Path: {OnnxModelPath}", modelSet.PipelineType.ToString(), modelSet.ResampleModelConfig.ExecutionProvider.Name, modelSet.ResampleModelConfig.OnnxModelPath);
        }


        /// <summary>
        /// Creates the pipeline from a ModelSet configuration.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static IPipeline CreatePipeline(StableDiffusionModelSet modelSet, ILogger logger = default)
        {
            return modelSet.PipelineType switch
            {
                PipelineType.StableDiffusionXL => StableDiffusionXLPipeline.CreatePipeline(modelSet, logger),
                PipelineType.LatentConsistency => LatentConsistencyPipeline.CreatePipeline(modelSet, logger),
                _ => StableDiffusionPipeline.CreatePipeline(modelSet, logger)
            };
        }
    }
}