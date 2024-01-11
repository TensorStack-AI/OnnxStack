using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.Core.Services;
using OnnxStack.Core.Video;
using OnnxStack.ImageUpscaler.Config;
using OnnxStack.ImageUpscaler.Extensions;
using OnnxStack.ImageUpscaler.Models;
using OnnxStack.StableDiffusion.Config;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.ImageUpscaler.Services
{
    public class UpscaleService : IUpscaleService
    {
        private readonly IOnnxModelService _modelService;
        private readonly ImageUpscalerConfig _configuration;
        private readonly IVideoService _videoService;

        /// <summary>
        /// Initializes a new instance of the <see cref="UpscaleService"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="modelService">The model service.</param>
        /// <param name="imageService">The image service.</param>
        public UpscaleService(ImageUpscalerConfig configuration, IOnnxModelService modelService, IVideoService videoService)
        {
            _configuration = configuration;
            _modelService = modelService;
            _videoService = videoService;
        }


        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        public async Task<bool> LoadModelAsync(UpscaleModelSet model)
        {
            var modelSet = await _modelService.LoadModelAsync(model);
            return modelSet is not null;
        }


        /// <summary>
        /// Unloads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        public async Task<bool> UnloadModelAsync(UpscaleModelSet model)
        {
            return await _modelService.UnloadModelAsync(model);
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
            return _modelService.IsModelLoaded(modelOptions);
        }


        /// <summary>
        /// Generates the upscaled image.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<DenseTensor<float>> GenerateAsync(UpscaleModelSet modelOptions, InputImage inputImage, CancellationToken cancellationToken = default)
        {
            return await GenerateInternalAsync(modelOptions, inputImage, cancellationToken);
        }


        /// <summary>
        /// Generates the upscaled image.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<Image<Rgba32>> GenerateAsImageAsync(UpscaleModelSet modelOptions, InputImage inputImage, CancellationToken cancellationToken = default)
        {
            var imageTensor = await GenerateInternalAsync(modelOptions, inputImage, cancellationToken);
            return imageTensor.ToImage(ImageNormalizeType.ZeroToOne);
        }


        /// <summary>
        /// Generates the upscaled image.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<byte[]> GenerateAsByteAsync(UpscaleModelSet modelOptions, InputImage inputImage, CancellationToken cancellationToken = default)
        {
            var imageTensor = await GenerateInternalAsync(modelOptions, inputImage, cancellationToken);
            using (var memoryStream = new MemoryStream())
            using (var image = imageTensor.ToImage(ImageNormalizeType.ZeroToOne))
            {
                await image.SaveAsPngAsync(memoryStream);
                return memoryStream.ToArray();
            }
        }


        /// <summary>
        /// Generates the upscaled image.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<Stream> GenerateAsStreamAsync(UpscaleModelSet modelOptions, InputImage inputImage, CancellationToken cancellationToken = default)
        {
            var imageTensor = await GenerateInternalAsync(modelOptions, inputImage, cancellationToken);
            using (var image = imageTensor.ToImage(ImageNormalizeType.ZeroToOne))
            {
                var memoryStream = new MemoryStream();
                await image.SaveAsPngAsync(memoryStream);
                return memoryStream;
            }
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
        /// Generates the upscaled video.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="videoInput">The video input.</param>
        /// <returns></returns>
        public async Task<byte[]> GenerateAsByteAsync(UpscaleModelSet modelOptions, VideoInput videoInput, CancellationToken cancellationToken)
        {
            var outputTasks = new List<Task<byte[]>>();
            var videoInfo = await _videoService.GetVideoInfoAsync(videoInput);
            var tensorFrames = await GenerateInternalAsync(modelOptions, videoInput, videoInfo, cancellationToken);
            foreach (DenseTensor<float> tensorFrame in tensorFrames)
            {
                outputTasks.Add(Task.Run(async () =>
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    using (var imageStream = new MemoryStream())
                    using (var imageFrame = tensorFrame.ToImage(ImageNormalizeType.ZeroToOne))
                    {
                        await imageFrame.SaveAsPngAsync(imageStream);
                        return imageStream.ToArray();
                    }
                }));
            }

            var output = await Task.WhenAll(outputTasks);
            var videoResult = await _videoService.CreateVideoAsync(output, videoInfo.FPS);
            return videoResult.Data;
        }


        /// <summary>
        /// Generates the upscaled video.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="videoInput">The video input.</param>
        /// <returns></returns>
        public async Task<Stream> GenerateAsStreamAsync(UpscaleModelSet modelOptions, VideoInput videoInput, CancellationToken cancellationToken)
        {
            return new MemoryStream(await GenerateAsByteAsync(modelOptions, videoInput, cancellationToken));
        }


        /// <summary>
        /// Generates an upscaled image of the source provided.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="inputImage">The input image.</param>
        private async Task<DenseTensor<float>> GenerateInternalAsync(UpscaleModelSet modelSet, InputImage inputImage, CancellationToken cancellationToken)
        {
            using (var image = await inputImage.ToImageAsync())
            {
                var upscaleInput = CreateInputParams(image, modelSet.SampleSize, modelSet.ScaleFactor);
                var metadata = _modelService.GetModelMetadata(modelSet, OnnxModelType.Upscaler);

                var outputTensor = new DenseTensor<float>(new[] { 1, modelSet.Channels, upscaleInput.OutputHeight, upscaleInput.OutputWidth });
                foreach (var imageTile in upscaleInput.ImageTiles)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    var outputDimension = new[] { 1, modelSet.Channels, imageTile.Destination.Height, imageTile.Destination.Width };
                    var inputTensor = imageTile.Image.ToDenseTensor(ImageNormalizeType.ZeroToOne, modelSet.Channels);
                    using (var inferenceParameters = new OnnxInferenceParameters(metadata))
                    {
                        inferenceParameters.AddInputTensor(inputTensor);
                        inferenceParameters.AddOutputBuffer(outputDimension);

                        var results = await _modelService.RunInferenceAsync(modelSet, OnnxModelType.Upscaler, inferenceParameters);
                        using (var result = results.First())
                        {
                            outputTensor.ApplyImageTile(result.ToDenseTensor(), imageTile.Destination);
                        }
                    }
                }

                return outputTensor;
            }
        }


        /// <summary>
        /// Generates the upscaled video.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="videoInput">The video input.</param>
        /// <returns></returns>
        public async Task<List<DenseTensor<float>>> GenerateInternalAsync(UpscaleModelSet modelOptions, VideoInput videoInput, VideoInfo videoInfo, CancellationToken cancellationToken)
        {
            var videoFrames = await _videoService.CreateFramesAsync(videoInput, videoInfo.FPS);
            var metadata = _modelService.GetModelMetadata(modelOptions, OnnxModelType.Upscaler);

            // Create Inputs
            var outputTensors = new List<DenseTensor<float>>();
            foreach (var frame in videoFrames.Frames)
            {
                using (var imageFrame = Image.Load<Rgba32>(frame))
                {
                    var input = CreateInputParams(imageFrame, modelOptions.SampleSize, modelOptions.ScaleFactor);
                    var outputDimension = new[] { 1, modelOptions.Channels, 0, 0 };
                    var outputTensor = new DenseTensor<float>(new[] { 1, modelOptions.Channels, input.OutputHeight, input.OutputWidth });
                    foreach (var imageTile in input.ImageTiles)
                    {
                        var inputTensor = imageTile.Image.ToDenseTensor(ImageNormalizeType.ZeroToOne, modelOptions.Channels);
                        outputDimension[2] = imageTile.Destination.Height;
                        outputDimension[3] = imageTile.Destination.Width;
                        using (var inferenceParameters = new OnnxInferenceParameters(metadata))
                        {
                            inferenceParameters.AddInputTensor(inputTensor);
                            inferenceParameters.AddOutputBuffer(outputDimension);

                            var results = await _modelService.RunInferenceAsync(modelOptions, OnnxModelType.Upscaler, inferenceParameters);
                            using (var result = results.First())
                            {
                                outputTensor.ApplyImageTile(result.ToDenseTensor(), imageTile.Destination);
                            }
                        }
                    }
                    outputTensors.Add(outputTensor);
                }
            }
            return outputTensors;
        }


        /// <summary>
        /// Creates the input parameters.
        /// </summary>
        /// <param name="imageSource">The image source.</param>
        /// <param name="maxTileSize">Maximum size of the tile.</param>
        /// <param name="scaleFactor">The scale factor.</param>
        /// <returns></returns>
        private static UpscaleInput CreateInputParams(Image<Rgba32> imageSource, int maxTileSize, int scaleFactor)
        {
            var tiles = imageSource.GenerateTiles(maxTileSize, scaleFactor);
            var width = imageSource.Width * scaleFactor;
            var height = imageSource.Height * scaleFactor;
            return new UpscaleInput(tiles, width, height);
        }
    }
}