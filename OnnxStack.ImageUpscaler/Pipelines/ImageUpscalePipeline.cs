using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.Core.Video;
using OnnxStack.ImageUpscaler.Common;
using System;
using System.Collections.Generic;
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
        public async Task LoadAsync()
        {
            await _upscaleModel.LoadAsync();
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
        public async Task<DenseTensor<float>> RunAsync(DenseTensor<float> inputImage, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Upscale DenseTensor..");
            var result = await UpscaleTensorAsync(inputImage, cancellationToken);
            _logger?.LogEnd("Upscale DenseTensor complete.", timestamp);
            return result;
        }


        /// <summary>
        /// Runs the upscale pipeline.
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async Task<OnnxImage> RunAsync(OnnxImage inputImage, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Upscale OnnxImage..");
            var result = await UpscaleImageAsync(inputImage, cancellationToken);
            _logger?.LogEnd("Upscale OnnxImage complete.", timestamp);
            return result;
        }


        /// <summary>
        /// Runs the pipline on a buffered video.
        /// </summary>
        /// <param name="inputVideo">The input video.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async Task<OnnxVideo> RunAsync(OnnxVideo inputVideo, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Upscale OnnxVideo..");
            var upscaledFrames = new List<OnnxImage>();
            foreach (var videoFrame in inputVideo.Frames)
            {
                upscaledFrames.Add(await UpscaleImageAsync(videoFrame, cancellationToken));
            }

            var firstFrame = upscaledFrames.First();
            var videoInfo = inputVideo.Info with
            {
                Width = firstFrame.Width,
                Height = firstFrame.Height,
            };

            _logger?.LogEnd("Upscale OnnxVideo complete.", timestamp);
            return new OnnxVideo(videoInfo, upscaledFrames);
        }


        /// <summary>
        /// Runs the pipline on a video stream.
        /// </summary>
        /// <param name="imageFrames">The image frames.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async IAsyncEnumerable<OnnxImage> RunAsync(IAsyncEnumerable<OnnxImage> imageFrames, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Upscale OnnxImage stream..");
            await foreach (var imageFrame in imageFrames)
            {
                yield return await UpscaleImageAsync(imageFrame, cancellationToken);
            }
            _logger?.LogEnd("Upscale OnnxImage stream complete.", timestamp);
        }


        /// <summary>
        /// Upscales the OnnxImage.
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        private async Task<OnnxImage> UpscaleImageAsync(OnnxImage inputImage, CancellationToken cancellationToken = default)
        {
            var inputTensor = inputImage.GetImageTensor(_upscaleModel.NormalizeType, _upscaleModel.Channels);
            var outputTensor = await RunInternalAsync(inputTensor, inputImage.Height, inputImage.Width, cancellationToken);
            return new OnnxImage(outputTensor, _upscaleModel.NormalizeType);
        }


        /// <summary>
        /// Upscales the DenseTensor
        /// </summary>
        /// <param name="inputTensor">The input Tensor.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async Task<DenseTensor<float>> UpscaleTensorAsync(DenseTensor<float> inputTensor, CancellationToken cancellationToken = default)
        {
            if (_upscaleModel.NormalizeInput && _upscaleModel.NormalizeType == ImageNormalizeType.ZeroToOne)
                inputTensor.NormalizeOneOneToZeroOne();

            var height = inputTensor.Dimensions[2];
            var width = inputTensor.Dimensions[3];
            var result = await RunInternalAsync(inputTensor, height, width, cancellationToken);

            if (_upscaleModel.NormalizeInput && _upscaleModel.NormalizeType == ImageNormalizeType.ZeroToOne)
                result.NormalizeZeroOneToOneOne();

            return result;
        }


        /// <summary>
        /// Runs the upscale pipeline
        /// </summary>
        /// <param name="inputTensor">The input tensor.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        private async Task<DenseTensor<float>> RunInternalAsync(DenseTensor<float> inputTensor, int height, int width, CancellationToken cancellationToken = default)
        {
            if (height <= _upscaleModel.TileSize && width <= _upscaleModel.TileSize)
            {
                return await RunInferenceAsync(inputTensor, cancellationToken);
            }

            var inputTiles = inputTensor.SplitImageTiles(_upscaleModel.TileOverlap);
            var outputTiles = new ImageTiles
            (
                inputTiles.Width * _upscaleModel.ScaleFactor,
                inputTiles.Height * _upscaleModel.ScaleFactor,
                inputTiles.Overlap * _upscaleModel.ScaleFactor,
                await RunInternalAsync(inputTiles.Tile1, inputTiles.Height, inputTiles.Width, cancellationToken),
                await RunInternalAsync(inputTiles.Tile2, inputTiles.Height, inputTiles.Width, cancellationToken),
                await RunInternalAsync(inputTiles.Tile3, inputTiles.Height, inputTiles.Width, cancellationToken),
                await RunInternalAsync(inputTiles.Tile4, inputTiles.Height, inputTiles.Width, cancellationToken)
            );
            return outputTiles.JoinImageTiles();
        }


        /// <summary>
        /// Runs the model inference.
        /// </summary>
        /// <param name="inputTensor">The input tensor.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        private async Task<DenseTensor<float>> RunInferenceAsync(DenseTensor<float> inputTensor, CancellationToken cancellationToken = default)
        {
            var metadata = await _upscaleModel.GetMetadataAsync();
            cancellationToken.ThrowIfCancellationRequested();

            var outputDimension = new[]
            {
               1,
               _upscaleModel.Channels,
               inputTensor.Dimensions[2] * _upscaleModel.ScaleFactor,
               inputTensor.Dimensions[3] * _upscaleModel.ScaleFactor
            };

            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(inputTensor);
                inferenceParameters.AddOutputBuffer(outputDimension);

                var results = await _upscaleModel.RunInferenceAsync(inferenceParameters);
                using (var result = results.First())
                {
                    return result.ToDenseTensor();
                }
            }
        }


        /// <summary>
        /// Creates the pipeline from a UpscaleModelSet.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static ImageUpscalePipeline CreatePipeline(UpscaleModelSet modelSet, ILogger logger = default)
        {
            var upscaleModel = new UpscaleModel(modelSet.UpscaleModelConfig.ApplyDefaults(modelSet));
            return new ImageUpscalePipeline(modelSet.Name, upscaleModel, logger);
        }


        /// <summary>
        /// Creates the pipeline from the specified folder.
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static ImageUpscalePipeline CreatePipeline(string modelFile, int scaleFactor, int sampleSize, ImageNormalizeType normalizeType = ImageNormalizeType.ZeroToOne, bool normalizeInput = true, int tileSize = 0, int tileOverlap = 20, int channels = 3, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML, ILogger logger = default)
        {
            var name = Path.GetFileNameWithoutExtension(modelFile);
            var configuration = new UpscaleModelSet
            {
                Name = name,
                IsEnabled = true,
                DeviceId = deviceId,
                ExecutionProvider = executionProvider,
                UpscaleModelConfig = new UpscaleModelConfig
                {
                    Channels = channels,
                    SampleSize = sampleSize,
                    ScaleFactor = scaleFactor,
                    TileOverlap = tileOverlap,
                    TileSize = Math.Min(sampleSize, tileSize > 0 ? tileSize : sampleSize),
                    NormalizeType = normalizeType,
                    NormalizeInput = normalizeInput,
                    OnnxModelPath = modelFile,
                }
            };
            return CreatePipeline(configuration, logger);
        }
    }

}
