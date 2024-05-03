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
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Pipelines
{
    public abstract class PipelineBase : IPipeline
    {
        protected readonly ILogger _logger;
        protected readonly PipelineOptions _pipelineOptions;

        /// <summary>
        /// Initializes a new instance of the <see cref="PipelineBase"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        protected PipelineBase(PipelineOptions pipelineOptions, ILogger logger)
        {
            _logger = logger;
            _pipelineOptions = pipelineOptions;
        }


        /// <summary>
        /// Gets the pipelines friendly name.
        /// </summary>
        public abstract string Name { get; }


        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public abstract DiffuserPipelineType PipelineType { get; }


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
        /// Loads the pipeline.
        /// </summary>
        /// <returns></returns>
        public abstract Task LoadAsync(bool controlNet = false);


        /// <summary>
        /// Unloads the pipeline.
        /// </summary>
        /// <returns></returns>
        public abstract Task UnloadAsync();


        /// <summary>
        /// Validates the inputs.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        public abstract void ValidateInputs(PromptOptions promptOptions, SchedulerOptions schedulerOptions);


        /// <summary>
        /// Runs the pipeline.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public abstract Task<DenseTensor<float>> RunAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);




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
        public abstract IAsyncEnumerable<BatchResult> RunBatchAsync(BatchOptions batchOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Runs the pipeline returning the result as an OnnxImage.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public abstract Task<OnnxImage> GenerateImageAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


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
        public abstract IAsyncEnumerable<BatchImageResult> GenerateImageBatchAsync(BatchOptions batchOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Runs the pipeline returning the result as an OnnxVideo.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public abstract Task<OnnxVideo> GenerateVideoAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


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
        public abstract IAsyncEnumerable<BatchVideoResult> GenerateVideoBatchAsync(BatchOptions batchOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Runs the video stream pipeline returning each frame as an OnnxImage.
        /// </summary>
        /// <param name="videoFrames">The video frames.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public abstract IAsyncEnumerable<OnnxImage> GenerateVideoStreamAsync(IAsyncEnumerable<OnnxImage> videoFrames, PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


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
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <param name="performGuidance">if set to <c>true</c> perform guidance (CFG).</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        protected async Task<DenseTensor<float>> DiffuseImageAsync(IDiffuser diffuser, PromptOptions promptOptions, SchedulerOptions schedulerOptions, PromptEmbeddingsResult promptEmbeddings, bool performGuidance, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var diffuseTime = _logger?.LogBegin("Image Diffuser starting...");
            var schedulerResult = await diffuser.DiffuseAsync(promptOptions, schedulerOptions, promptEmbeddings, performGuidance, progressCallback, cancellationToken);
            _logger?.LogEnd($"Image Diffuser complete", diffuseTime);
            return schedulerResult;
        }


        /// <summary>
        /// Runs the Diffusion process and returns a video.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <param name="performGuidance">if set to <c>true</c> perform guidance (CFG).</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        protected async IAsyncEnumerable<DenseTensor<float>> DiffuseVideoAsync(IDiffuser diffuser, PromptOptions promptOptions, SchedulerOptions schedulerOptions, PromptEmbeddingsResult promptEmbeddings, bool performGuidance, Action<DiffusionProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var diffuseTime = _logger?.LogBegin("Video Diffuser starting...");

            var frameIndex = 0;
            var videoFrames = promptOptions.InputVideo.Frames;
            foreach (var videoFrame in videoFrames)
            {
                if (promptOptions.DiffuserType == DiffuserType.ControlNet || promptOptions.DiffuserType == DiffuserType.ControlNetImage)
                {
                    // ControlNetImage uses frame as input image
                    if (promptOptions.DiffuserType == DiffuserType.ControlNetImage)
                        promptOptions.InputImage = videoFrame;

                    promptOptions.InputContolImage = promptOptions.InputContolVideo?.GetFrame(frameIndex);
                }
                else
                {
                    promptOptions.InputImage = videoFrame;
                }

                var frameResultTensor = await diffuser.DiffuseAsync(promptOptions, schedulerOptions, promptEmbeddings, performGuidance, progressCallback, cancellationToken);

                // Frame Progress
                ReportBatchProgress(progressCallback, ++frameIndex, videoFrames.Count, frameResultTensor);

                // Concatenate frame
                yield return frameResultTensor;
            }

            _logger?.LogEnd($"Video Diffuser complete", diffuseTime);
        }


        /// <summary>
        /// Check if we should run guidance.
        /// </summary>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <returns></returns>
        protected virtual bool ShouldPerformGuidance(SchedulerOptions schedulerOptions)
        {
            return schedulerOptions.GuidanceScale > 1f;
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
        protected void ReportBatchProgress(Action<DiffusionProgress> progressCallback, int progress, int progressMax, DenseTensor<float> progressTensor)
        {
            progressCallback?.Invoke(new DiffusionProgress
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
        protected Action<DiffusionProgress> CreateBatchCallback(Action<DiffusionProgress> progressCallback, int batchCount, Func<int> batchIndex)
        {
            if (progressCallback == null)
                return progressCallback;

            return (DiffusionProgress progress) => progressCallback?.Invoke(new DiffusionProgress
            {
                StepMax = progress.StepMax,
                StepValue = progress.StepValue,
                StepTensor = progress.StepTensor,
                BatchMax = batchCount,
                BatchValue = batchIndex(),
                BatchTensor = progress.BatchTensor,
                Message = progress.Message
            });
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
                DiffuserPipelineType.StableDiffusionXL => StableDiffusionXLPipeline.CreatePipeline(modelSet, logger),
                DiffuserPipelineType.LatentConsistency => LatentConsistencyPipeline.CreatePipeline(modelSet, logger),
                DiffuserPipelineType.LatentConsistencyXL => LatentConsistencyXLPipeline.CreatePipeline(modelSet, logger),
                DiffuserPipelineType.InstaFlow => InstaFlowPipeline.CreatePipeline(modelSet, logger),
                _ => StableDiffusionPipeline.CreatePipeline(modelSet, logger)
            };
        }
    }
}