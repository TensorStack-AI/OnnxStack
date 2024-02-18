using Microsoft.Extensions.Logging;
using OnnxStack.Core.Config;
using OnnxStack.Core.Image;
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
        private readonly ILogger<UpscaleService> _logger;
        private readonly Dictionary<IOnnxModel, ImageUpscalePipeline> _pipelines;

        /// <summary>
        /// Initializes a new instance of the <see cref="UpscaleService"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="modelService">The model service.</param>
        /// <param name="imageService">The image service.</param>
        public UpscaleService(ILogger<UpscaleService> logger)
        {
            _logger = logger;
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
        /// <param name="modelSet">The model options.</param>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<OnnxImage> GenerateAsync(UpscaleModelSet modelSet, OnnxImage inputImage, CancellationToken cancellationToken = default)
        {
            if (!_pipelines.TryGetValue(modelSet, out var pipeline))
                throw new Exception("Pipeline not found or is unsupported");

            return await pipeline.RunAsync(inputImage, cancellationToken);
        }


        /// <summary>
        /// Generates the upscaled video.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="inputVideo">The input video.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        /// <exception cref="System.Exception">Pipeline not found or is unsupported</exception>
        public async Task<OnnxVideo> GenerateAsync(UpscaleModelSet modelSet, OnnxVideo inputVideo, CancellationToken cancellationToken = default)
        {
            if (!_pipelines.TryGetValue(modelSet, out var pipeline))
                throw new Exception("Pipeline not found or is unsupported");

            return await pipeline.RunAsync(inputVideo, cancellationToken);
        }


        /// <summary>
        /// Creates the pipeline.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <returns></returns>
        private ImageUpscalePipeline CreatePipeline(UpscaleModelSet modelSet)
        {
            return ImageUpscalePipeline.CreatePipeline(modelSet, _logger);
        }
    }
}