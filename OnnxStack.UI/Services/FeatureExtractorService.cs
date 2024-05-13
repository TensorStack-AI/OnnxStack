using Microsoft.Extensions.Logging;
using OnnxStack.Core.Config;
using OnnxStack.Core.Image;
using OnnxStack.Core.Video;
using OnnxStack.FeatureExtractor.Common;
using OnnxStack.FeatureExtractor.Pipelines;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.UI.Services
{
    public class FeatureExtractorService : IFeatureExtractorService
    {
        private readonly ILogger<FeatureExtractorService> _logger;
        private readonly Dictionary<IOnnxModel, FeatureExtractorPipeline> _pipelines;

        /// <summary>
        /// Initializes a new instance of the <see cref="FeatureExtractorService"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="modelService">The model service.</param>
        /// <param name="imageService">The image service.</param>
        public FeatureExtractorService(ILogger<FeatureExtractorService> logger)
        {
            _logger = logger;
            _pipelines = new Dictionary<IOnnxModel, FeatureExtractorPipeline>();
        }


        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        public async Task<bool> LoadModelAsync(FeatureExtractorModelSet model)
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
        public async Task<bool> UnloadModelAsync(FeatureExtractorModelSet model)
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
        /// <param name="model">The model.</param>
        /// <returns>
        ///   <c>true</c> if [is model loaded] [the specified model options]; otherwise, <c>false</c>.
        /// </returns>
        /// <exception cref="System.NotImplementedException"></exception>
        public bool IsModelLoaded(FeatureExtractorModelSet model)
        {
            return _pipelines.ContainsKey(model);
        }


        /// <summary>
        /// Generates the feature image.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<OnnxImage> GenerateAsync(FeatureExtractorModelSet model, OnnxImage inputImage, CancellationToken cancellationToken = default)
        {
            if (!_pipelines.TryGetValue(model, out var pipeline))
                throw new Exception("Pipeline not found or is unsupported");

            return await pipeline.RunAsync(inputImage, cancellationToken);
        }


        /// <summary>
        /// Generates the feature video.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="inputVideo">The input video.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        /// <exception cref="System.Exception">Pipeline not found or is unsupported</exception>
        public async Task<OnnxVideo> GenerateAsync(FeatureExtractorModelSet model, OnnxVideo inputVideo, CancellationToken cancellationToken = default)
        {
            if (!_pipelines.TryGetValue(model, out var pipeline))
                throw new Exception("Pipeline not found or is unsupported");

            return await pipeline.RunAsync(inputVideo, cancellationToken: cancellationToken);
        }


        /// <summary>
        /// Creates the pipeline.
        /// </summary>
        /// <param name="modelSet">The model.</param>
        /// <returns></returns>
        private FeatureExtractorPipeline CreatePipeline(FeatureExtractorModelSet model)
        {
            return FeatureExtractorPipeline.CreatePipeline(model, _logger);
        }
    }
}