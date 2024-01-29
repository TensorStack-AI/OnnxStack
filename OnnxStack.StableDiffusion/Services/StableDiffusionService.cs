using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Image;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.StableDiffusion.Pipelines;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Services
{
    /// <summary>
    /// Service for generating images using text and image based prompts
    /// </summary>
    /// <seealso cref="OnnxStack.StableDiffusion.Common.IStableDiffusionService" />
    public sealed class StableDiffusionService : IStableDiffusionService
    {
        private readonly IVideoService _videoService;
        private readonly ILogger<StableDiffusionService> _logger;
        private readonly StableDiffusionConfig _configuration;
        private readonly Dictionary<IOnnxModel, IPipeline> _pipelines;
        private readonly ConcurrentDictionary<IOnnxModel, ControlNetModel> _controlNetSessions;

        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionService"/> class.
        /// </summary>
        /// <param name="schedulerService">The scheduler service.</param>
        public StableDiffusionService(StableDiffusionConfig configuration, IVideoService videoService, ILogger<StableDiffusionService> logger)
        {
            _logger = logger;
            _configuration = configuration;
            _videoService = videoService;
            _pipelines = new Dictionary<IOnnxModel, IPipeline>();
            _controlNetSessions = new ConcurrentDictionary<IOnnxModel, ControlNetModel>();
        }


        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        public async Task<bool> LoadModelAsync(StableDiffusionModelSet model)
        {
            if (_pipelines.ContainsKey(model))
                return true;

            var pipeline = CreatePipeline(model);
            await pipeline.LoadAsync();
            return _pipelines.TryAdd(model, pipeline);
        }




        /// <summary>
        /// Unloads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        public async Task<bool> UnloadModelAsync(StableDiffusionModelSet model)
        {
            if (_pipelines.Remove(model, out var pipeline))
            {
                await pipeline?.UnloadAsync();
            }
            return true;
        }


        /// <summary>
        /// Determines whether [is model loaded] [the specified model options].
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <returns>
        ///   <c>true</c> if [is model loaded] [the specified model options]; otherwise, <c>false</c>.
        /// </returns>
        /// <exception cref="System.NotImplementedException"></exception>
        public bool IsModelLoaded(StableDiffusionModelSet modelOptions)
        {
            return _pipelines.ContainsKey(modelOptions);
        }


        public async Task<bool> LoadControlNetModelAsync(ControlNetModelSet model)
        {
            if (_controlNetSessions.ContainsKey(model))
                return true;

            var config = model.ControlNetConfig.ApplyDefaults(model);
            var controlNet = new ControlNetModel(config);
            await controlNet.LoadAsync();
            return _controlNetSessions.TryAdd(model, controlNet);
        }

        public Task<bool> UnloadControlNetModelAsync(ControlNetModelSet model)
        {
            if (_controlNetSessions.Remove(model, out var controlNet))
            {
                controlNet?.UnloadAsync();
            }
            return Task.FromResult(true);
        }

        public bool IsControlNetModelLoaded(ControlNetModelSet modelOptions)
        {
            return _controlNetSessions.ContainsKey(modelOptions);
        }


        /// <summary>
        /// Generates the StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The Scheduler options.</param>
        /// <param name="progressCallback">The callback used to provide progess of the current InferenceSteps.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>The diffusion result as <see cref="DenseTensor<float>"/></returns>
        public async Task<DenseTensor<float>> GenerateAsync(ModelOptions model, PromptOptions prompt, SchedulerOptions options, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            return await DiffuseAsync(model, prompt, options, progressCallback, cancellationToken).ConfigureAwait(false);
        }


        /// <summary>
        /// Generates the StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The Scheduler options.</param>
        /// <param name="progressCallback">The callback used to provide progess of the current InferenceSteps.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>The diffusion result as <see cref="SixLabors.ImageSharp.Image<Rgba32>"/></returns>
        public async Task<Image<Rgba32>> GenerateAsImageAsync(ModelOptions model, PromptOptions prompt, SchedulerOptions options, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            return await GenerateAsync(model, prompt, options, progressCallback, cancellationToken)
                .ContinueWith(t => t.Result.ToImage(), cancellationToken)
                .ConfigureAwait(false);
        }


        /// <summary>
        /// Generates the StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The Scheduler options.</param>
        /// <param name="progressCallback">The callback used to provide progess of the current InferenceSteps.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>The diffusion result as <see cref="byte[]"/></returns>
        public async Task<byte[]> GenerateAsBytesAsync(ModelOptions model, PromptOptions prompt, SchedulerOptions options, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var generateResult = await GenerateAsync(model, prompt, options, progressCallback, cancellationToken).ConfigureAwait(false);
            if (!prompt.HasInputVideo)
                return generateResult.ToImageBytes();

            return await GenerateVideoResultAsBytesAsync(generateResult, prompt.VideoOutputFPS, progressCallback, cancellationToken).ConfigureAwait(false);
        }


        /// <summary>
        /// Generates the StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The Scheduler options.</param>
        /// <param name="progressCallback">The callback used to provide progess of the current InferenceSteps.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>The diffusion result as <see cref="System.IO.Stream"/></returns>
        public async Task<Stream> GenerateAsStreamAsync(ModelOptions model, PromptOptions prompt, SchedulerOptions options, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var generateResult = await GenerateAsync(model, prompt, options, progressCallback, cancellationToken).ConfigureAwait(false);
            if (!prompt.HasInputVideo)
                return generateResult.ToImageStream();

            return await GenerateVideoResultAsStreamAsync(generateResult, prompt.VideoOutputFPS, progressCallback, cancellationToken).ConfigureAwait(false);
        }


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
        public IAsyncEnumerable<BatchResult> GenerateBatchAsync(ModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            return DiffuseBatchAsync(modelOptions, promptOptions, schedulerOptions, batchOptions, progressCallback, cancellationToken);
        }


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
        public async IAsyncEnumerable<Image<Rgba32>> GenerateBatchAsImageAsync(ModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<DiffusionProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await foreach (var result in GenerateBatchAsync(modelOptions, promptOptions, schedulerOptions, batchOptions, progressCallback, cancellationToken))
                yield return result.ImageResult.ToImage();
        }


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
        public async IAsyncEnumerable<byte[]> GenerateBatchAsBytesAsync(ModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<DiffusionProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await foreach (var result in GenerateBatchAsync(modelOptions, promptOptions, schedulerOptions, batchOptions, progressCallback, cancellationToken))
            {
                if (!promptOptions.HasInputVideo)
                    yield return result.ImageResult.ToImageBytes();

                yield return await GenerateVideoResultAsBytesAsync(result.ImageResult, promptOptions.VideoOutputFPS, progressCallback, cancellationToken).ConfigureAwait(false);
            }
        }


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
        public async IAsyncEnumerable<Stream> GenerateBatchAsStreamAsync(ModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<DiffusionProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await foreach (var result in GenerateBatchAsync(modelOptions, promptOptions, schedulerOptions, batchOptions, progressCallback, cancellationToken))
            {
                if (!promptOptions.HasInputVideo)
                    yield return result.ImageResult.ToImageStream();

                yield return await GenerateVideoResultAsStreamAsync(result.ImageResult, promptOptions.VideoOutputFPS, progressCallback, cancellationToken).ConfigureAwait(false);
            }
        }


        /// <summary>
        /// Runs the diffusion process
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="progress">The progress.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        /// <exception cref="System.Exception">
        /// Pipeline not found or is unsupported
        /// or
        /// Diffuser not found or is unsupported
        /// or
        /// Scheduler '{schedulerOptions.SchedulerType}' is not compatible  with the `{pipeline.PipelineType}` pipeline.
        /// </exception>
        private async Task<DenseTensor<float>> DiffuseAsync(ModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, Action<DiffusionProgress> progress = null, CancellationToken cancellationToken = default)
        {
            if (!_pipelines.TryGetValue(modelOptions.BaseModel, out var pipeline))
                throw new Exception("Pipeline not found or is unsupported");

            var controlNet = default(ControlNetModel);
            if (modelOptions.ControlNetModel is not null && !_controlNetSessions.TryGetValue(modelOptions.ControlNetModel, out controlNet))
                throw new Exception("ControlNet not loaded");

            pipeline.ValidateInputs(promptOptions, schedulerOptions);

            await GenerateInputVideoFrames(promptOptions, progress);
            return await pipeline.RunAsync(promptOptions, schedulerOptions, controlNet, progress, cancellationToken);
        }


        /// <summary>
        /// Runs the batch diffusion process.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="batchOptions">The batch options.</param>
        /// <param name="progress">The progress.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        /// <exception cref="System.Exception">
        /// Pipeline not found or is unsupported
        /// or
        /// Diffuser not found or is unsupported
        /// or
        /// Scheduler '{schedulerOptions.SchedulerType}' is not compatible  with the `{pipeline.PipelineType}` pipeline.
        /// </exception>
        private async IAsyncEnumerable<BatchResult> DiffuseBatchAsync(ModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<DiffusionProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            if (!_pipelines.TryGetValue(modelOptions.BaseModel, out var pipeline))
                throw new Exception("Pipeline not found or is unsupported");

            var controlNet = default(ControlNetModel);
            if (modelOptions.ControlNetModel is not null && !_controlNetSessions.TryGetValue(modelOptions.ControlNetModel, out controlNet))
                throw new Exception("ControlNet not loaded");

            pipeline.ValidateInputs(promptOptions, schedulerOptions);

            await GenerateInputVideoFrames(promptOptions, progressCallback);
            await foreach (var result in pipeline.RunBatchAsync(promptOptions, schedulerOptions, batchOptions, controlNet, progressCallback, cancellationToken))
            {
                yield return result;
            }
        }


        /// <summary>
        /// Generates the video result as bytes.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="videoTensor">The video tensor.</param>
        /// <param name="progress">The progress.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        private async Task<byte[]> GenerateVideoResultAsBytesAsync(DenseTensor<float> videoTensor, float videoFPS, Action<DiffusionProgress> progress = null, CancellationToken cancellationToken = default)
        {
            progress?.Invoke(new DiffusionProgress("Generating Video Result..."));
            var videoResult = await _videoService.CreateVideoAsync(videoTensor, videoFPS, cancellationToken);
            return videoResult.Data;
        }


        /// <summary>
        /// Generates the video result as stream.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="videoTensor">The video tensor.</param>
        /// <param name="progress">The progress.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        private async Task<MemoryStream> GenerateVideoResultAsStreamAsync(DenseTensor<float> videoTensor, float videoFPS, Action<DiffusionProgress> progress = null, CancellationToken cancellationToken = default)
        {
            return new MemoryStream(await GenerateVideoResultAsBytesAsync(videoTensor, videoFPS, progress, cancellationToken));
        }


        /// <summary>
        /// Generates the input video frames.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="progress">The progress.</param>
        private async Task GenerateInputVideoFrames(PromptOptions promptOptions, Action<DiffusionProgress> progress)
        {
            if (!promptOptions.HasInputVideo || promptOptions.InputVideo.VideoFrames is not null)
                return;

            if (promptOptions.VideoInputFPS == 0 || promptOptions.VideoOutputFPS == 0)
            {
                var videoInfo = await _videoService.GetVideoInfoAsync(promptOptions.InputVideo);
                if (promptOptions.VideoInputFPS == 0)
                    promptOptions.VideoInputFPS = videoInfo.FPS;

                if (promptOptions.VideoOutputFPS == 0)
                    promptOptions.VideoOutputFPS = videoInfo.FPS;
            }

            var videoFrame = await _videoService.CreateFramesAsync(promptOptions.InputVideo, promptOptions.VideoInputFPS);
            progress?.Invoke(new DiffusionProgress($"Generating video frames @ {promptOptions.VideoInputFPS}fps"));
            promptOptions.InputVideo.VideoFrames = videoFrame;
        }


        private IPipeline CreatePipeline(StableDiffusionModelSet model)
        {
            return model.PipelineType switch
            {
                DiffuserPipelineType.StableDiffusion => StableDiffusionPipeline.CreatePipeline(model, _logger),
                DiffuserPipelineType.StableDiffusionXL => StableDiffusionXLPipeline.CreatePipeline(model, _logger),
                DiffuserPipelineType.LatentConsistency => LatentConsistencyPipeline.CreatePipeline(model, _logger),
                DiffuserPipelineType.LatentConsistencyXL => LatentConsistencyXLPipeline.CreatePipeline(model, _logger),
                DiffuserPipelineType.InstaFlow => InstaFlowPipeline.CreatePipeline(model, _logger),
                _ => throw new NotSupportedException()
            }; ;
        }
    }
}
