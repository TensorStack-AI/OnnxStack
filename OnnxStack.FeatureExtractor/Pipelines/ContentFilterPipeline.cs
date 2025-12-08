using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.Core.Video;
using OnnxStack.FeatureExtractor.Common;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.FeatureExtractor.Pipelines
{
    public class ContentFilterPipeline
    {
        private readonly string _name;
        private readonly ILogger _logger;
        private readonly FeatureExtractorModel _featureExtractorModel;

        /// <summary>
        /// Initializes a new instance of the <see cref="FeatureExtractorPipeline"/> class.
        /// </summary>
        /// <param name="name">The name.</param>
        /// <param name="featureExtractorModel">The feature exxtractor model.</param>
        /// <param name="logger">The logger.</param>
        public ContentFilterPipeline(string name, FeatureExtractorModel featureExtractorModel, ILogger logger = default)
        {
            _name = name;
            _logger = logger;
            _featureExtractorModel = featureExtractorModel;
        }


        /// <summary>
        /// Gets the name.
        /// </summary>
        /// <value>
        public string Name => _name;


        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <returns></returns>
        public Task LoadAsync(CancellationToken cancellationToken = default)
        {
            return _featureExtractorModel.LoadAsync(cancellationToken: cancellationToken);
        }


        /// <summary>
        /// Unloads the models.
        /// </summary>
        public async Task UnloadAsync()
        {
            await Task.Yield();
            _featureExtractorModel?.Dispose();
        }


        /// <summary>
        /// Checks if the image contains expicit content
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;System.Boolean&gt; representing the asynchronous operation.</returns>
        public async Task<bool> ContainsExpicitContentAsync(OnnxImage inputImage, float threshold = 0.02f, bool isLowMemoryEnabled = false, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Extracting OnnxImage feature...");
            var filterContent = await RunInternalAsync(inputImage, threshold, cancellationToken);
            if (isLowMemoryEnabled)
                await _featureExtractorModel.UnloadAsync();

            _logger?.LogEnd("Extracting OnnxImage feature complete.", timestamp);
            return filterContent;
        }


        /// <summary>
        /// Generates the feature extractor image
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<DenseTensor<float>> RunAsync(DenseTensor<float> inputTensor, float threshold = 0.02f, bool isLowMemoryEnabled = false, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Extracting DenseTensor feature...");
            var inputImage = new OnnxImage(inputTensor);
            var filterContent = await RunInternalAsync(inputImage, threshold, cancellationToken);
            var result = filterContent ? CensorImage(inputImage) : inputImage;
            if (isLowMemoryEnabled)
                await _featureExtractorModel.UnloadAsync();

            _logger?.LogEnd("Extracting DenseTensor feature complete.", timestamp);
            return await result.GetImageTensorAsync();
        }


        /// <summary>
        /// Generates the feature extractor image
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<OnnxImage> RunAsync(OnnxImage inputImage, float threshold = 0.02f, bool isLowMemoryEnabled = false, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Extracting OnnxImage feature...");
            var filterContent = await RunInternalAsync(inputImage, threshold, cancellationToken);
            if (isLowMemoryEnabled)
                await _featureExtractorModel.UnloadAsync();

            _logger?.LogEnd("Extracting OnnxImage feature complete.", timestamp);
            return filterContent ? CensorImage(inputImage) : inputImage;
        }


        /// <summary>
        /// Generates the feature extractor video
        /// </summary>
        /// <param name="videoFrames">The input video.</param>
        /// <returns></returns>
        public async Task<OnnxVideo> RunAsync(OnnxVideo video, float threshold = 0.02f, bool isLowMemoryEnabled = false, Action<OnnxImage, OnnxImage> progressCallback = default, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Extracting OnnxVideo features...");
            var featureFrames = new List<OnnxImage>();
            foreach (var videoFrame in video.Frames)
            {
                var imageResult = videoFrame;
                var filterContent = await RunInternalAsync(videoFrame, threshold, cancellationToken);
                if (filterContent)
                    imageResult = CensorImage(videoFrame);

                featureFrames.Add(imageResult);
                progressCallback?.Invoke(videoFrame, imageResult);
            }

            if (isLowMemoryEnabled)
                await _featureExtractorModel.UnloadAsync();

            _logger?.LogEnd("Extracting OnnxVideo features complete.", timestamp);
            return new OnnxVideo(featureFrames, video.FrameRate);
        }


        /// <summary>
        /// Generates the feature extractor video stream
        /// </summary>
        /// <param name="imageFrames">The image frames.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async IAsyncEnumerable<OnnxImage> RunAsync(IAsyncEnumerable<OnnxImage> imageFrames, float threshold = 0.02f, bool isLowMemoryEnabled = false, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Extracting OnnxImage stream features...");
            await foreach (var imageFrame in imageFrames)
            {
                var filterContent = await RunInternalAsync(imageFrame, threshold, cancellationToken);
                if (filterContent)
                    yield return CensorImage(imageFrame);

                yield return imageFrame;
            }

            if (isLowMemoryEnabled)
                await _featureExtractorModel.UnloadAsync();

            _logger?.LogEnd("Extracting OnnxImage stream features complete.", timestamp);
        }


        /// <summary>
        /// Runs the pipeline
        /// </summary>
        /// <param name="inputTensor">The input tensor.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        private async Task<bool> RunInternalAsync(OnnxImage image, float threshold = 0.02f, CancellationToken cancellationToken = default)
        {
            var metadata = await _featureExtractorModel.LoadAsync(cancellationToken: cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();

            var inputTensor = image.GetClipImageFeatureTensor();
            using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
            {
                inferenceParameters.AddInputTensor(inputTensor);
                inferenceParameters.AddOutputBuffer([1, 17]);
                var results = _featureExtractorModel.RunInference(inferenceParameters);
                using (var result = results[0])
                {
                    var concepts = result.ToDenseTensor();
                    return concepts.Any(x => x > threshold);
                }
            }
        }


        private static OnnxImage CensorImage(OnnxImage onnxImage)
        {
            onnxImage.GetImage().Mutate(c => c.GaussianBlur(50));
            return onnxImage;
        }


        /// <summary>
        /// Creates the pipeline from a FeatureExtractorModelConfig.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static ContentFilterPipeline CreatePipeline(FeatureExtractorModelConfig configuration, ILogger logger = default)
        {
            var featureExtractorModel = new FeatureExtractorModel(configuration);
            return new ContentFilterPipeline(configuration.Name, featureExtractorModel, logger);
        }


        public static ContentFilterPipeline CreatePipeline(OnnxExecutionProvider executionProvider, string modelFile, ILogger logger = default)
        {
            var name = Path.GetFileNameWithoutExtension(modelFile);
            var configuration = new FeatureExtractorModelConfig
            {
                Name = name,
                OnnxModelPath = modelFile,
                ExecutionProvider = executionProvider,
            };
            return CreatePipeline(configuration, logger);
        }
    }
}
