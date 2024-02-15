using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.ImageUpscaler.Common;
using OnnxStack.ImageUpscaler.Extensions;
using OnnxStack.ImageUpscaler.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.IO;
using System.Linq;
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
        public async Task<DenseTensor<float>> RunAsync(OnnxImage inputImage, CancellationToken cancellationToken = default)
        {
            var upscaleInput = CreateInputParams(inputImage, _upscaleModel.SampleSize, _upscaleModel.ScaleFactor);
            var metadata = await _upscaleModel.GetMetadataAsync();

            var outputTensor = new DenseTensor<float>(new[] { 1, _upscaleModel.Channels, upscaleInput.OutputHeight, upscaleInput.OutputWidth });
            foreach (var imageTile in upscaleInput.ImageTiles)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var outputDimension = new[] { 1, _upscaleModel.Channels, imageTile.Destination.Height, imageTile.Destination.Width };
                var inputTensor = imageTile.Image.GetImageTensor(ImageNormalizeType.ZeroToOne, _upscaleModel.Channels);
                using (var inferenceParameters = new OnnxInferenceParameters(metadata))
                {
                    inferenceParameters.AddInputTensor(inputTensor);
                    inferenceParameters.AddOutputBuffer(outputDimension);

                    var results = await _upscaleModel.RunInferenceAsync(inferenceParameters);
                    using (var result = results.First())
                    {
                        outputTensor.ApplyImageTile(result.ToDenseTensor(), imageTile.Destination);
                    }
                }
            }
            return outputTensor;
        }


        private static UpscaleInput CreateInputParams(OnnxImage imageSource, int maxTileSize, int scaleFactor)
        {
            var tiles = imageSource.GenerateTiles(maxTileSize, scaleFactor);
            var width = imageSource.Width * scaleFactor;
            var height = imageSource.Height * scaleFactor;
            return new UpscaleInput(tiles, width, height);
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
        public static ImageUpscalePipeline CreatePipeline(string modelFile, int scaleFactor, int sampleSize = 512, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML, ILogger logger = default)
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
                    Channels = 3,
                    SampleSize = sampleSize,
                    ScaleFactor = scaleFactor,
                    OnnxModelPath = modelFile
                }
            };
            return CreatePipeline(configuration, logger);
        }
    }
}
