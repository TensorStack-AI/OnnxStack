using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.Core.Video;
using OnnxStack.FeatureExtractor.Common;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
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
        public virtual Task LoadAsync(CancellationToken cancellationToken = default)
        {
            return _featureExtractorModel.LoadAsync(cancellationToken: cancellationToken);
        }


        /// <summary>
        /// Unloads the models.
        /// </summary>
        public virtual async Task UnloadAsync()
        {
            await Task.Yield();
            _featureExtractorModel?.Dispose();
        }


        /// <summary>
        /// Generates the feature extractor image
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<DenseTensor<float>> RunAsync(DenseTensor<float> inputTensor, FeatureExtractorOptions options, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Extracting DenseTensor feature...");
            var result = await ExtractTensorAsync(inputTensor, options, cancellationToken);
            if (options.IsLowMemoryEnabled)
                await UnloadAsync();

            _logger?.LogEnd("Extracting DenseTensor feature complete.", timestamp);
            return result;
        }


        /// <summary>
        /// Generates the feature extractor image
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<OnnxImage> RunAsync(OnnxImage inputImage, FeatureExtractorOptions options, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Extracting OnnxImage feature...");
            var result = await ExtractImageAsync(inputImage, options, cancellationToken);
            if (options.IsLowMemoryEnabled)
                await UnloadAsync();

            _logger?.LogEnd("Extracting OnnxImage feature complete.", timestamp);
            return result;
        }


        /// <summary>
        /// Generates the feature extractor video
        /// </summary>
        /// <param name="videoFrames">The input video.</param>
        /// <returns></returns>
        public async Task<OnnxVideo> RunAsync(OnnxVideo video, FeatureExtractorOptions options, IProgress<FeatureExtractorProgress> progressCallback = default, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Extracting OnnxVideo features...");
            var featureFrames = new List<OnnxImage>();
            foreach (var videoFrame in video.Frames)
            {
                var frameTime = Stopwatch.GetTimestamp();
                var result = await ExtractImageAsync(videoFrame, options, cancellationToken);
                featureFrames.Add(result);
                progressCallback?.Report(new FeatureExtractorProgress(videoFrame, result, Stopwatch.GetElapsedTime(frameTime).TotalMilliseconds));
            }

            if (options.IsLowMemoryEnabled)
                await UnloadAsync();

            _logger?.LogEnd("Extracting OnnxVideo features complete.", timestamp);
            return new OnnxVideo(featureFrames, video.FrameRate);
        }


        /// <summary>
        /// Generates the feature extractor video stream
        /// </summary>
        /// <param name="imageFrames">The image frames.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async IAsyncEnumerable<OnnxImage> RunAsync(IAsyncEnumerable<OnnxImage> imageFrames, FeatureExtractorOptions options, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Extracting OnnxImage stream features...");
            await foreach (var imageFrame in imageFrames)
            {
                yield return await ExtractImageAsync(imageFrame, options, cancellationToken);
            }

            if (options.IsLowMemoryEnabled)
                await UnloadAsync();

            _logger?.LogEnd("Extracting OnnxImage stream features complete.", timestamp);
        }


        /// <summary>
        /// Extracts the feature to OnnxImage.
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        protected virtual async Task<OnnxImage> ExtractImageAsync(OnnxImage inputImage, FeatureExtractorOptions options, CancellationToken cancellationToken = default)
        {
            var inputTensor = await inputImage.GetImageTensorAsync();
            var outputTensor = await ExtractTensorAsync(inputTensor, options, cancellationToken);
            return new OnnxImage(outputTensor);
        }


        /// <summary>
        /// Extracts the feature to DenseTensor.
        /// </summary>
        /// <param name="inputTensor">The input tensor.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        protected virtual async Task<DenseTensor<float>> ExtractTensorAsync(DenseTensor<float> imageTensor, FeatureExtractorOptions options, CancellationToken cancellationToken = default)
        {
            if (options.TileMode == TileMode.None)
            {
                if (_featureExtractorModel.SampleSize > 0)
                {
                    var originalHeight = imageTensor.Dimensions[2];
                    var originalWidth = imageTensor.Dimensions[3];
                    var (scaledWidth, scaledHeight) = ScaleToSampleSize(originalWidth, originalHeight, _featureExtractorModel.SampleSize);
                    var isRescaleRequired = originalHeight != scaledWidth || originalWidth != scaledWidth;
                    if (isRescaleRequired)
                        imageTensor = imageTensor.ResizeImage(scaledWidth, scaledHeight);
                    imageTensor = await ExecuteExtractorAsync(imageTensor, cancellationToken);
                    if (isRescaleRequired)
                        return imageTensor.ResizeImage(originalWidth, originalHeight);

                    return imageTensor;
                }
                return await ExecuteExtractorAsync(imageTensor, cancellationToken);
            }

            return await ExecuteExtractorTilesAsync(imageTensor, options.MaxTileSize, options.TileMode, options.TileOverlap, cancellationToken);
        }


        /// <summary>
        /// Execute extractor.
        /// </summary>
        /// <param name="inputTensor">The input tensor.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;DenseTensor`1&gt; representing the asynchronous operation.</returns>
        private async Task<DenseTensor<float>> ExecuteExtractorAsync(DenseTensor<float> inputTensor, CancellationToken cancellationToken = default)
        {
            var originalHeight = inputTensor.Dimensions[2];
            var originalWidth = inputTensor.Dimensions[3];

            var metadata = await _featureExtractorModel.LoadAsync(cancellationToken: cancellationToken);
            var outputShape = new[] { 1, _featureExtractorModel.OutputChannels, inputTensor.Dimensions[2], inputTensor.Dimensions[3] };

            cancellationToken.ThrowIfCancellationRequested();
            using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
            {
                cancellationToken.ThrowIfCancellationRequested();

                if (_featureExtractorModel.NormalizeType == ImageNormalizeType.ZeroToOne)
                    inputTensor.NormalizeOneOneToZeroOne();

                inferenceParameters.AddInputTensor(inputTensor);
                inferenceParameters.AddOutputBuffer();

                if (_featureExtractorModel.NormalizeType == ImageNormalizeType.ZeroToOne)
                    inputTensor.NormalizeZeroOneToOneOne();

                var inferenceResults = _featureExtractorModel.RunInference(inferenceParameters);
                using (var inferenceResult = inferenceResults[0])
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    var outputTensor = inferenceResult.ToDenseTensor();
                    if (outputTensor.Dimensions.Length == 3)
                        outputTensor = outputTensor.ReshapeTensor([1, .. outputTensor.Dimensions]);

                    if (outputTensor.Dimensions[2] != originalHeight || outputTensor.Dimensions[3] != originalWidth)
                        outputTensor = outputTensor.ResizeImage(originalWidth, originalHeight);

                    if (_featureExtractorModel.InvertOutput)
                        InvertOutput(outputTensor);

                    if (_featureExtractorModel.NormalizeOutputType == ImageNormalizeType.MinMax)
                        outputTensor.NormalizeMinMax();
                    else if (_featureExtractorModel.NormalizeOutputType == ImageNormalizeType.OneToOne)
                        outputTensor.NormalizeZeroOneToOneOne();

                    if (_featureExtractorModel.SetOutputToInputAlpha)
                        return AddAlphaChannel(inputTensor, outputTensor);

                    return outputTensor;
                }
            }
        }


        /// <summary>
        /// Execute extractor as tiles
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="maxTileSize">Maximum size of the tile.</param>
        /// <param name="tileMode">The tile mode.</param>
        /// <param name="tileOverlap">The tile overlap.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;DenseTensor`1&gt; representing the asynchronous operation.</returns>
        private async Task<DenseTensor<float>> ExecuteExtractorTilesAsync(DenseTensor<float> imageTensor, int maxTileSize, TileMode tileMode, int tileOverlap, CancellationToken cancellationToken = default)
        {
            if (_featureExtractorModel.SampleSize > 0)
                maxTileSize = _featureExtractorModel.SampleSize - tileOverlap;

            var height = imageTensor.Dimensions[2];
            var width = imageTensor.Dimensions[3];
            if (width <= (maxTileSize + tileOverlap) || height <= (maxTileSize + tileOverlap))
                return await ExecuteExtractorAsync(imageTensor, cancellationToken);

            var inputTiles = new ImageTiles(imageTensor, tileMode, tileOverlap);
            var outputTiles = new ImageTiles
            (
                inputTiles.Width,
                inputTiles.Height,
                tileMode,
                inputTiles.Overlap,
                await ExecuteExtractorTilesAsync(inputTiles.Tile1, maxTileSize, tileMode, tileOverlap, cancellationToken),
                await ExecuteExtractorTilesAsync(inputTiles.Tile2, maxTileSize, tileMode, tileOverlap, cancellationToken),
                await ExecuteExtractorTilesAsync(inputTiles.Tile3, maxTileSize, tileMode, tileOverlap, cancellationToken),
                await ExecuteExtractorTilesAsync(inputTiles.Tile4, maxTileSize, tileMode, tileOverlap, cancellationToken)
            );
            return outputTiles.JoinTiles();
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
        /// Inverts the output.
        /// </summary>
        /// <param name="values">The values.</param>
        private static void InvertOutput(DenseTensor<float> values)
        {
            for (int j = 0; j < values.Length; j++)
            {
                values.SetValue(j, -values.GetValue(j));
            }
        }


        /// <summary>
        /// Scales the input to the nreaest SampleSize.
        /// </summary>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="sampleSize">Size of the sample.</param>
        /// <returns>System.ValueTuple&lt;System.Int32, System.Int32&gt;.</returns>
        private static (int newWidth, int newHeight) ScaleToSampleSize(int width, int height, int sampleSize)
        {
            if (width <= sampleSize && height <= sampleSize)
                return (width, height);

            float scale = (width < height) ? (float)sampleSize / width : (float)sampleSize / height;
            int newWidth = (int)(width * scale);
            int newHeight = (int)(height * scale);
            return (newWidth, newHeight);
        }


        /// <summary>
        /// Creates the pipeline from a FeatureExtractorModelSet.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static FeatureExtractorPipeline CreatePipeline(FeatureExtractorModelConfig configuration, ILogger logger = default)
        {
            var featureExtractorModel = new FeatureExtractorModel(configuration);
            return new FeatureExtractorPipeline(configuration.Name, featureExtractorModel, logger);
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
        public static FeatureExtractorPipeline CreatePipeline(OnnxExecutionProvider executionProvider, string modelFile, int sampleSize = 0, int outputChannels = 1, ImageNormalizeType normalizeType = ImageNormalizeType.None, ImageNormalizeType normalizeOutputType = ImageNormalizeType.None, ImageResizeMode inputResizeMode = ImageResizeMode.Crop, bool setOutputToInputAlpha = false, bool invertOutput = false, ILogger logger = default)
        {
            var name = Path.GetFileNameWithoutExtension(modelFile);
            var configuration = new FeatureExtractorModelConfig
            {
                Name = name,
                OnnxModelPath = modelFile,
                ExecutionProvider = executionProvider,
                SampleSize = sampleSize,
                OutputChannels = outputChannels,
                NormalizeType = normalizeType,
                NormalizeOutputType = normalizeOutputType,
                SetOutputToInputAlpha = setOutputToInputAlpha,
                InputResizeMode = inputResizeMode,
                InvertOutput = invertOutput
            };
            return CreatePipeline(configuration, logger);
        }
    }
}
