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
        /// Loads the model.
        /// </summary>
        /// <returns></returns>
        public Task LoadAsync()
        {
            return _featureExtractorModel.LoadAsync();
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
        /// Generates the feature extractor image
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<DenseTensor<float>> RunAsync(DenseTensor<float> inputTensor, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Extracting DenseTensor feature...");
            var result = await ExtractTensorAsync(inputTensor, cancellationToken);
            _logger?.LogEnd("Extracting DenseTensor feature complete.", timestamp);
            return result;
        }


        /// <summary>
        /// Generates the feature extractor image
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<OnnxImage> RunAsync(OnnxImage inputImage, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Extracting OnnxImage feature...");
            var result = await ExtractImageAsync(inputImage, cancellationToken);
            _logger?.LogEnd("Extracting OnnxImage feature complete.", timestamp);
            return result;
        }


        /// <summary>
        /// Generates the feature extractor video
        /// </summary>
        /// <param name="videoFrames">The input video.</param>
        /// <returns></returns>
        public async Task<OnnxVideo> RunAsync(OnnxVideo video, Action<OnnxImage, OnnxImage> progressCallback = default, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Extracting OnnxVideo features...");
            var featureFrames = new List<OnnxImage>();
            foreach (var videoFrame in video.Frames)
            {
                var result = await ExtractImageAsync(videoFrame, cancellationToken);
                featureFrames.Add(result);
                progressCallback?.Invoke(videoFrame, result);
            }

            _logger?.LogEnd("Extracting OnnxVideo features complete.", timestamp);
            return new OnnxVideo(video.Info, featureFrames);
        }


        /// <summary>
        /// Generates the feature extractor video stream
        /// </summary>
        /// <param name="imageFrames">The image frames.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async IAsyncEnumerable<OnnxImage> RunAsync(IAsyncEnumerable<OnnxImage> imageFrames, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Extracting OnnxImage stream features...");
            await foreach (var imageFrame in imageFrames)
            {
                yield return await ExtractImageAsync(imageFrame, cancellationToken);
            }
            _logger?.LogEnd("Extracting OnnxImage stream features complete.", timestamp);
        }


        /// <summary>
        /// Extracts the feature to OnnxImage.
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        private async Task<OnnxImage> ExtractImageAsync(OnnxImage inputImage, CancellationToken cancellationToken = default)
        {
            var originalWidth = inputImage.Width;
            var originalHeight = inputImage.Height;
            var inputTensor = _featureExtractorModel.SampleSize <= 0
                ? await inputImage.GetImageTensorAsync(_featureExtractorModel.NormalizeType)
                : await inputImage.GetImageTensorAsync(_featureExtractorModel.SampleSize, _featureExtractorModel.SampleSize, _featureExtractorModel.NormalizeType, resizeMode: _featureExtractorModel.InputResizeMode);

            var outputTensor = await RunInternalAsync(inputTensor, cancellationToken);
            var imageResult = new OnnxImage(outputTensor, _featureExtractorModel.NormalizeType);

            if (_featureExtractorModel.InputResizeMode == ImageResizeMode.Stretch && (imageResult.Width != originalWidth || imageResult.Height != originalHeight))
                imageResult.Resize(originalHeight, originalWidth, _featureExtractorModel.InputResizeMode);

            return imageResult;
        }


        /// <summary>
        /// Extracts the feature to DenseTensor.
        /// </summary>
        /// <param name="inputTensor">The input tensor.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async Task<DenseTensor<float>> ExtractTensorAsync(DenseTensor<float> inputTensor, CancellationToken cancellationToken = default)
        {
            if (_featureExtractorModel.NormalizeInput && _featureExtractorModel.NormalizeType == ImageNormalizeType.ZeroToOne)
                inputTensor.NormalizeOneOneToZeroOne();

            return await RunInternalAsync(inputTensor, cancellationToken); 
        }


        /// <summary>
        /// Runs the pipeline
        /// </summary>
        /// <param name="inputTensor">The input tensor.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        private async Task<DenseTensor<float>> RunInternalAsync(DenseTensor<float> inputTensor, CancellationToken cancellationToken = default)
        {
            var metadata = await _featureExtractorModel.GetMetadataAsync();
            cancellationToken.ThrowIfCancellationRequested();
            var outputShape = new[] { 1, _featureExtractorModel.OutputChannels, inputTensor.Dimensions[2], inputTensor.Dimensions[3] };
            var outputBuffer = metadata.Outputs[0].Value.Dimensions.Length == 4 ? outputShape : outputShape[1..];
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(inputTensor);
                inferenceParameters.AddOutputBuffer(outputBuffer);

                var inferenceResults = await _featureExtractorModel.RunInferenceAsync(inferenceParameters);
                using (var inferenceResult = inferenceResults.First())
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    var outputTensor = inferenceResult.ToDenseTensor(outputShape);
                    if (_featureExtractorModel.NormalizeOutput)
                        outputTensor.NormalizeMinMax();

                    if (_featureExtractorModel.SetOutputToInputAlpha)
                        return AddAlphaChannel(inputTensor, outputTensor);

                    return outputTensor;
                }
            }
        }


        /// <summary>
        /// Adds an alpha channel to the RGB tensor.
        /// </summary>
        /// <param name="sourceImage">The source image.</param>
        /// <param name="alphaChannel">The alpha channel.</param>
        /// <returns></returns>
        private static DenseTensor<float> AddAlphaChannel(DenseTensor<float> sourceImage, DenseTensor<float> alphaChannel)
        {
            var resultTensor = new DenseTensor<float>(new int[] { 1, 4, sourceImage.Dimensions[2], sourceImage.Dimensions[3] });
            sourceImage.Buffer.Span.CopyTo(resultTensor.Buffer[..(int)sourceImage.Length].Span);
            alphaChannel.Buffer.Span.CopyTo(resultTensor.Buffer[(int)sourceImage.Length..].Span);
            return resultTensor;
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
        /// Creates the pipeline from the specified arguments.
        /// </summary>
        /// <param name="modelFile">The model file.</param>
        /// <param name="sampleSize">Size of the sample.</param>
        /// <param name="channels">The channels.</param>
        /// <param name="normalizeOutputTensor">if set to <c>true</c> [normalize output tensor].</param>
        /// <param name="normalizeInputTensor">The normalize input tensor.</param>
        /// <param name="setOutputToInputAlpha">if set to <c>true</c> [set output to input alpha].</param>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static FeatureExtractorPipeline CreatePipeline(string modelFile, int sampleSize = 0, int outputChannels = 1, ImageNormalizeType normalizeType = ImageNormalizeType.ZeroToOne, bool normalizeInput = true, bool normalizeOutput = false, ImageResizeMode inputResizeMode = ImageResizeMode.Crop, bool setOutputToInputAlpha = false, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML, ILogger logger = default)
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
                    OutputChannels = outputChannels,
                    NormalizeOutput = normalizeOutput,
                    NormalizeInput = normalizeInput,
                    NormalizeType = normalizeType,
                    SetOutputToInputAlpha = setOutputToInputAlpha,
                    InputResizeMode = inputResizeMode
                }
            };
            return CreatePipeline(configuration, logger);
        }
    }
}
