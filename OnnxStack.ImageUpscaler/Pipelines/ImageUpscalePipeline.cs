using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.Core.Video;
using OnnxStack.ImageUpscaler.Common;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.FeatureExtractor.Pipelines
{
    public class ImageUpscalePipeline
    {
        private readonly string _name;
        private readonly ILogger _logger;
        private readonly UpscaleModel _upscaleModel;


        /// <summary>
        /// Initializes a new instance of the <see cref="ImageUpscalePipeline"/> class.
        /// </summary>
        /// <param name="name">The name.</param>
        /// <param name="upscaleModel">The upscale model.</param>
        /// <param name="logger">The logger.</param>
        public ImageUpscalePipeline(string name, UpscaleModel upscaleModel, ILogger logger = default)
        {
            _name = name;
            _logger = logger;
            _upscaleModel = upscaleModel;
        }


        /// <summary>
        /// Gets the name.
        /// </summary>
        /// <value>
        public string Name => _name;


        /// <summary>
        /// Loads the model.
        /// </summary>
        public async Task LoadAsync(CancellationToken cancellationToken = default)
        {
            await _upscaleModel.LoadAsync(cancellationToken: cancellationToken);
        }


        /// <summary>
        /// Unloads the models.
        /// </summary>
        public async Task UnloadAsync()
        {
            await Task.Yield();
            _upscaleModel?.Dispose();
        }


        /// <summary>
        /// Runs the upscale pipeline.
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async Task<DenseTensor<float>> RunAsync(DenseTensor<float> inputImage, UpscaleOptions options, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Upscale DenseTensor..");
            var result = await UpscaleTensorAsync(inputImage, options, cancellationToken);
            if (options.IsLowMemoryEnabled)
                await _upscaleModel.UnloadAsync();

            _logger?.LogEnd("Upscale DenseTensor complete.", timestamp);
            return result;
        }


        /// <summary>
        /// Runs the upscale pipeline.
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async Task<OnnxImage> RunAsync(OnnxImage inputImage, UpscaleOptions options, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Upscale OnnxImage..");
            var result = await UpscaleImageAsync(inputImage, options, cancellationToken);
            if (options.IsLowMemoryEnabled)
                await _upscaleModel.UnloadAsync();

            _logger?.LogEnd("Upscale OnnxImage complete.", timestamp);
            return result;
        }


        /// <summary>
        /// Runs the pipline on a buffered video.
        /// </summary>
        /// <param name="inputVideo">The input video.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async Task<OnnxVideo> RunAsync(OnnxVideo inputVideo, UpscaleOptions options, IProgress<UpscaleProgress> progressCallback = default, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Upscale OnnxVideo..");
            var upscaledFrames = new List<OnnxImage>();
            foreach (var videoFrame in inputVideo.Frames)
            {
                var frameTime = Stopwatch.GetTimestamp();
                var result = await UpscaleImageAsync(videoFrame, options, cancellationToken);
                upscaledFrames.Add(result);
                progressCallback?.Report(new UpscaleProgress(videoFrame, result, Stopwatch.GetElapsedTime(frameTime).TotalMilliseconds));
            }

            if (options.IsLowMemoryEnabled)
                await _upscaleModel.UnloadAsync();

            _logger?.LogEnd("Upscale OnnxVideo complete.", timestamp);
            return new OnnxVideo(upscaledFrames, inputVideo.FrameRate);
        }


        /// <summary>
        /// Runs the pipline on a video stream.
        /// </summary>
        /// <param name="imageFrames">The image frames.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async IAsyncEnumerable<OnnxImage> RunAsync(IAsyncEnumerable<OnnxImage> imageFrames, UpscaleOptions options, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Upscale OnnxImage stream..");
            await foreach (var imageFrame in imageFrames)
            {
                yield return await UpscaleImageAsync(imageFrame, options, cancellationToken);
            }

            if (options.IsLowMemoryEnabled)
                await _upscaleModel.UnloadAsync();

            _logger?.LogEnd("Upscale OnnxImage stream complete.", timestamp);
        }


        /// <summary>
        /// Upscales the OnnxImage.
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        private async Task<OnnxImage> UpscaleImageAsync(OnnxImage inputImage, UpscaleOptions options, CancellationToken cancellationToken = default)
        {
            var inputTensor = inputImage.GetImageTensor(channels: _upscaleModel.Channels);
            var outputTensor = await UpscaleTensorAsync(inputTensor, options, cancellationToken);
            return new OnnxImage(outputTensor);
        }


        /// <summary>
        /// Upscales the DenseTensor
        /// </summary>
        /// <param name="inputTensor">The input Tensor.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async Task<DenseTensor<float>> UpscaleTensorAsync(DenseTensor<float> inputTensor, UpscaleOptions options, CancellationToken cancellationToken = default)
        {
            if (_upscaleModel.NormalizeType == ImageNormalizeType.ZeroToOne)
                inputTensor.NormalizeOneOneToZeroOne();

            var result = await UpscaleInternalAsync(inputTensor, options, cancellationToken);
            if (_upscaleModel.NormalizeType == ImageNormalizeType.ZeroToOne)
            {
                inputTensor.NormalizeZeroOneToOneOne();
                result.NormalizeZeroOneToOneOne();
            }

            return result;
        }


        private async Task<DenseTensor<float>> UpscaleInternalAsync(DenseTensor<float> imageTensor, UpscaleOptions options, CancellationToken cancellationToken = default)
        {
            return options.TileMode == TileMode.None
                ? await ExecuteUpscaleAsync(imageTensor, cancellationToken)
                : await ExecuteUpscaleTilesAsync(imageTensor, options.MaxTileSize, options.TileMode, options.TileOverlap, cancellationToken);
        }


        private async Task<DenseTensor<float>> ExecuteUpscaleAsync(DenseTensor<float> imageTensor, CancellationToken cancellationToken = default)
        {
            var height = imageTensor.Dimensions[2];
            var width = imageTensor.Dimensions[3];
            var metadata = await _upscaleModel.LoadAsync(cancellationToken: cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();
            var outputDimension = new[] { 1, _upscaleModel.Channels, height * _upscaleModel.ScaleFactor, width * _upscaleModel.ScaleFactor };
            using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
            {
                inferenceParameters.AddInputTensor(imageTensor);
                inferenceParameters.AddOutputBuffer(outputDimension);

                var results = await _upscaleModel.RunInferenceAsync(inferenceParameters);
                using (var result = results.First())
                {
                    return result.ToDenseTensor();
                }
            }
        }


        private async Task<DenseTensor<float>> ExecuteUpscaleTilesAsync(DenseTensor<float> imageTensor, int maxTileSize, TileMode tileMode, int tileOverlap, CancellationToken cancellationToken = default)
        {
            if (_upscaleModel.SampleSize > 0)
                maxTileSize = _upscaleModel.SampleSize - tileOverlap;

            var height = imageTensor.Dimensions[2];
            var width = imageTensor.Dimensions[3];
            if (width <= (maxTileSize + tileOverlap) || height <= (maxTileSize + tileOverlap))
                return await ExecuteUpscaleAsync(imageTensor, cancellationToken);

            var inputTiles = new ImageTiles(imageTensor, tileMode, tileOverlap);
            var outputTiles = new ImageTiles
            (
                inputTiles.Width * _upscaleModel.ScaleFactor,
                inputTiles.Height * _upscaleModel.ScaleFactor,
                tileMode,
                inputTiles.Overlap * _upscaleModel.ScaleFactor,
                await ExecuteUpscaleTilesAsync(inputTiles.Tile1, maxTileSize, tileMode, tileOverlap, cancellationToken),
                await ExecuteUpscaleTilesAsync(inputTiles.Tile2, maxTileSize, tileMode, tileOverlap, cancellationToken),
                await ExecuteUpscaleTilesAsync(inputTiles.Tile3, maxTileSize, tileMode, tileOverlap, cancellationToken),
                await ExecuteUpscaleTilesAsync(inputTiles.Tile4, maxTileSize, tileMode, tileOverlap, cancellationToken)
            );
            return outputTiles.JoinTiles();
        }


        /// <summary>
        /// Creates the pipeline from a UpscaleModelSet.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static ImageUpscalePipeline CreatePipeline(UpscaleModelConfig configuration, ILogger logger = default)
        {
            var upscaleModel = new UpscaleModel(configuration);
            return new ImageUpscalePipeline(configuration.Name, upscaleModel, logger);
        }


        /// <summary>
        /// Creates the pipeline from the specified folder.
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static ImageUpscalePipeline CreatePipeline(OnnxExecutionProvider executionProvider, string modelFile, int scaleFactor, int sampleSize, ImageNormalizeType normalizeType = ImageNormalizeType.ZeroToOne, bool normalizeInput = true, int tileSize = 0, int tileOverlap = 20, int channels = 3, ILogger logger = default)
        {
            var name = Path.GetFileNameWithoutExtension(modelFile);
            var configuration = new UpscaleModelConfig
            {
                Name = name,
                OnnxModelPath = modelFile,
                ExecutionProvider = executionProvider,
                Channels = channels,
                SampleSize = sampleSize,
                ScaleFactor = scaleFactor,
                NormalizeType = normalizeType
            };
            return CreatePipeline(configuration, logger);
        }
    }

}
