using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Image;
using OnnxStack.Core.Services;
using OnnxStack.Core.Video;
using OnnxStack.FeatureExtractor.Pipelines;
using OnnxStack.ImageUpscaler.Common;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.UI.Services
{
    public class UpscaleService : IUpscaleService
    {
        private readonly IVideoService _videoService;
        private readonly ILogger<StableDiffusionService> _logger;
        private readonly Dictionary<IOnnxModel, ImageUpscalePipeline> _pipelines;

        /// <summary>
        /// Initializes a new instance of the <see cref="UpscaleService"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="modelService">The model service.</param>
        /// <param name="imageService">The image service.</param>
        public UpscaleService(IVideoService videoService)
        {
            _videoService = videoService;
            _pipelines = new Dictionary<IOnnxModel, ImageUpscalePipeline>();
        }


        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        public async Task<bool> LoadModelAsync(UpscaleModelSet model)
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
        public async Task<bool> UnloadModelAsync(UpscaleModelSet model)
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
        public bool IsModelLoaded(UpscaleModelSet modelOptions)
        {
            return _pipelines.ContainsKey(modelOptions);
        }


        /// <summary>
        /// Generates the upscaled image.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<OnnxImage> GenerateAsync(UpscaleModelSet modelOptions, OnnxImage inputImage, CancellationToken cancellationToken = default)
        {
            return new OnnxImage(await GenerateInternalAsync(modelOptions, inputImage, cancellationToken), ImageNormalizeType.ZeroToOne);
        }




        /// <summary>
        /// Generates the upscaled video.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="videoInput">The video input.</param>
        /// <returns></returns>
        public async Task<DenseTensor<float>> GenerateAsync(UpscaleModelSet modelOptions, VideoInput videoInput, CancellationToken cancellationToken = default)
        {
            var videoInfo = await _videoService.GetVideoInfoAsync(videoInput);
            var tensorFrames = await GenerateInternalAsync(modelOptions, videoInput, videoInfo, cancellationToken);

            DenseTensor<float> videoResult = default;
            foreach (var tensorFrame in tensorFrames)
            {
                cancellationToken.ThrowIfCancellationRequested();
                videoResult = videoResult.Concatenate(tensorFrame);
            }
            return videoResult;
        }


        /// <summary>
        /// Generates an upscaled image of the source provided.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="inputImage">The input image.</param>
        private async Task<DenseTensor<float>> GenerateInternalAsync(UpscaleModelSet modelSet, OnnxImage inputImage, CancellationToken cancellationToken)
        {
            if (!_pipelines.TryGetValue(modelSet, out var pipeline))
                throw new Exception("Pipeline not found or is unsupported");

            return await pipeline.RunAsync(inputImage, cancellationToken);
        }


        /// <summary>
        /// Generates the upscaled video.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="videoInput">The video input.</param>
        /// <returns></returns>
        public async Task<List<DenseTensor<float>>> GenerateInternalAsync(UpscaleModelSet modelSet, VideoInput videoInput, VideoInfo videoInfo, CancellationToken cancellationToken)
        {
            if (!_pipelines.TryGetValue(modelSet, out var pipeline))
                throw new Exception("Pipeline not found or is unsupported");

            return new List<DenseTensor<float>>();
        }





        private ImageUpscalePipeline CreatePipeline(UpscaleModelSet modelSet)
        {
            return ImageUpscalePipeline.CreatePipeline(modelSet, _logger);
        }

    }
}