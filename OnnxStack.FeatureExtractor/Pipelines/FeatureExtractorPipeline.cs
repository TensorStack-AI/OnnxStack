using Microsoft.Extensions.Logging;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.Core.Video;
using OnnxStack.FeatureExtractor.Common;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.FeatureExtractor.Pipelines
{
    public class FeatureExtractorPipeline
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
        public FeatureExtractorPipeline(string name, FeatureExtractorModel featureExtractorModel, ILogger logger = default)
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
        /// Unloads the models.
        /// </summary>
        public async Task UnloadAsync()
        {
            await Task.Yield();
            _featureExtractorModel?.Dispose();
        }


        /// <summary>
        /// Generates the feature extractor image
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<InputImage> RunAsync(InputImage inputImage, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Extracting image feature...");
            var controlImage = await inputImage.ToDenseTensorAsync(_featureExtractorModel.SampleSize, _featureExtractorModel.SampleSize, ImageNormalizeType.ZeroToOne);
            var metadata = await _featureExtractorModel.GetMetadataAsync();
            cancellationToken.ThrowIfCancellationRequested();
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(controlImage);
                inferenceParameters.AddOutputBuffer(new[] { 1, _featureExtractorModel.Channels, _featureExtractorModel.SampleSize, _featureExtractorModel.SampleSize });

                var results = await _featureExtractorModel.RunInferenceAsync(inferenceParameters);
                using (var result = results.First())
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    var resultTensor = result.ToDenseTensor();
                    if (_featureExtractorModel.Normalize)
                        resultTensor.NormalizeMinMax();

                    var maskImage = resultTensor.ToImageMask();
                    //await maskImage.SaveAsPngAsync("D:\\Mask.png");
                    _logger?.LogEnd("Extracting image feature complete.", timestamp);
                    return new InputImage(maskImage);
                }
            }
        }


        /// <summary>
        /// Generates the feature extractor video
        /// </summary>
        /// <param name="videoFrames">The input video.</param>
        /// <returns></returns>
        public async Task<VideoFrames> RunAsync(VideoFrames videoFrames, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Extracting video features...");
            var metadata = await _featureExtractorModel.GetMetadataAsync();
            cancellationToken.ThrowIfCancellationRequested();

            foreach (var videoFrame in videoFrames.Frames)
            {
                var image = new InputImage(videoFrame.Frame);
                var controlImage = await image.ToDenseTensorAsync(_featureExtractorModel.SampleSize, _featureExtractorModel.SampleSize, ImageNormalizeType.ZeroToOne);
                using (var inferenceParameters = new OnnxInferenceParameters(metadata))
                {
                    inferenceParameters.AddInputTensor(controlImage);
                    inferenceParameters.AddOutputBuffer(new[] { 1, _featureExtractorModel.Channels, _featureExtractorModel.SampleSize, _featureExtractorModel.SampleSize });

                    var results = await _featureExtractorModel.RunInferenceAsync(inferenceParameters);
                    using (var result = results.First())
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        var resultTensor = result.ToDenseTensor();
                        if (_featureExtractorModel.Normalize)
                            resultTensor.NormalizeMinMax();

                        var maskImage = resultTensor.ToImageMask();
                        videoFrame.ExtraFrame = new InputImage(maskImage);
                    }
                }
            }
            _logger?.LogEnd("Extracting video features complete.", timestamp);
            return videoFrames;
        }


        /// <summary>
        /// Creates the pipeline from a FeatureExtractorModelSet.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static FeatureExtractorPipeline CreatePipeline(FeatureExtractorModelSet modelSet, ILogger logger = default)
        {
            var featureExtractorModel = new FeatureExtractorModel(modelSet.FeatureExtractorConfig.ApplyDefaults(modelSet));
            return new FeatureExtractorPipeline(modelSet.Name, featureExtractorModel, logger);
        }


        /// <summary>
        /// Creates the pipeline from the specified file.
        /// </summary>
        /// <param name="modelFile">The model file.</param>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static FeatureExtractorPipeline CreatePipeline(string modelFile, bool normalize = false, int sampleSize = 512, int channels = 1, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML, ILogger logger = default)
        {
            var name = Path.GetFileNameWithoutExtension(modelFile);
            var configuration = new FeatureExtractorModelSet
            {
                Name = name,
                IsEnabled = true,
                DeviceId = deviceId,
                ExecutionProvider = executionProvider,
                FeatureExtractorConfig = new FeatureExtractorModelConfig
                {
                    OnnxModelPath = modelFile,
                    SampleSize = sampleSize,
                    Normalize = normalize,
                    Channels = channels
                }
            };
            return CreatePipeline(configuration, logger);
        }
    }
}
