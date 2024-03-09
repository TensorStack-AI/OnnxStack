using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.Core.Video;
using OnnxStack.FeatureExtractor.Common;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.FeatureExtractor.Pipelines
{
    public class BackgroundRemovalPipeline
    {
        private readonly string _name;
        private readonly ILogger _logger;
        private readonly FeatureExtractorModel _model;

        /// <summary>
        /// Initializes a new instance of the <see cref="BackgroundRemovalPipeline"/> class.
        /// </summary>
        /// <param name="name">The name.</param>
        /// <param name="model">The model.</param>
        /// <param name="logger">The logger.</param>
        public BackgroundRemovalPipeline(string name, FeatureExtractorModel model, ILogger logger = default)
        {
            _name = name;
            _logger = logger;
            _model = model;
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
        public Task LoadAsync()
        {
            return _model.LoadAsync();
        }


        /// <summary>
        /// Unloads the models.
        /// </summary>
        public async Task UnloadAsync()
        {
            await Task.Yield();
            _model?.Dispose();
        }


        /// <summary>
        /// Generates the background removal image result
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<OnnxImage> RunAsync(OnnxImage inputImage, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Removing video background...");
            var result = await RunInternalAsync(inputImage, cancellationToken);
            _logger?.LogEnd("Removing video background complete.", timestamp);
            return result;
        }


        /// <summary>
        /// Generates the background removal video result
        /// </summary>
        /// <param name="videoFrames">The input video.</param>
        /// <returns></returns>
        public async Task<OnnxVideo> RunAsync(OnnxVideo video, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Removing video background...");
            var videoFrames = new List<OnnxImage>();
            foreach (var videoFrame in video.Frames)
            {
                videoFrames.Add(await RunInternalAsync(videoFrame, cancellationToken));
            }
            _logger?.LogEnd("Removing video background complete.", timestamp);
            return new OnnxVideo(video.Info with
            {
                Height = videoFrames[0].Height,
                Width = videoFrames[0].Width,
            }, videoFrames);
        }


        /// <summary>
        /// Generates the background removal video stream
        /// </summary>
        /// <param name="imageFrames">The image frames.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async IAsyncEnumerable<OnnxImage> RunAsync(IAsyncEnumerable<OnnxImage> imageFrames, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Extracting video stream features...");
            await foreach (var imageFrame in imageFrames)
            {
                yield return await RunInternalAsync(imageFrame, cancellationToken);
            }
            _logger?.LogEnd("Extracting video stream features complete.", timestamp);
        }


        /// <summary>
        /// Runs the pipeline
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        private async Task<OnnxImage> RunInternalAsync(OnnxImage inputImage, CancellationToken cancellationToken = default)
        {
            var souceImageTenssor = await inputImage.GetImageTensorAsync(_model.SampleSize, _model.SampleSize, ImageNormalizeType.ZeroToOne);
            var metadata = await _model.GetMetadataAsync();
            cancellationToken.ThrowIfCancellationRequested();
            var outputShape = new[] { 1, _model.Channels, _model.SampleSize, _model.SampleSize };
            var outputBuffer = metadata.Outputs[0].Value.Dimensions.Length == 4 ? outputShape : outputShape[1..];
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(souceImageTenssor);
                inferenceParameters.AddOutputBuffer(outputBuffer);

                var results = await _model.RunInferenceAsync(inferenceParameters);
                using (var result = results.First())
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    var imageTensor = AddAlphaChannel(souceImageTenssor, result.GetTensorDataAsSpan<float>());
                    return new OnnxImage(imageTensor, ImageNormalizeType.ZeroToOne);
                }
            }
        }


        /// <summary>
        /// Adds an alpha channel to the RGB tensor.
        /// </summary>
        /// <param name="sourceImage">The source image.</param>
        /// <param name="alphaChannel">The alpha channel.</param>
        /// <returns></returns>
        private static DenseTensor<float> AddAlphaChannel(DenseTensor<float> sourceImage, ReadOnlySpan<float> alphaChannel)
        {
            var resultTensor = new DenseTensor<float>(new int[] { 1, 4, sourceImage.Dimensions[2], sourceImage.Dimensions[3] });
            sourceImage.Buffer.Span.CopyTo(resultTensor.Buffer[..(int)sourceImage.Length].Span);
            alphaChannel.CopyTo(resultTensor.Buffer[(int)sourceImage.Length..].Span);
            return resultTensor;
        }


        /// <summary>
        /// Creates the pipeline from a FeatureExtractorModelSet.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static BackgroundRemovalPipeline CreatePipeline(FeatureExtractorModelSet modelSet, ILogger logger = default)
        {
            var model = new FeatureExtractorModel(modelSet.FeatureExtractorConfig.ApplyDefaults(modelSet));
            return new BackgroundRemovalPipeline(modelSet.Name, model, logger);
        }


        /// <summary>
        /// Creates the pipeline from the specified file.
        /// </summary>
        /// <param name="modelFile">The model file.</param>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static BackgroundRemovalPipeline CreatePipeline(string modelFile, int sampleSize = 512, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML, ILogger logger = default)
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
                    Normalize = false,
                    Channels = 1
                }
            };
            return CreatePipeline(configuration, logger);
        }
    }
}
