using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using OnnxStack.Core.Services;
using OnnxStack.ImageUpscaler.Config;
using OnnxStack.ImageUpscaler.Extensions;
using OnnxStack.ImageUpscaler.Models;
using OnnxStack.StableDiffusion.Config;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace OnnxStack.ImageUpscaler.Services
{
    public class UpscaleService : IUpscaleService
    {
        private readonly IImageService _imageService;
        private readonly IOnnxModelService _modelService;
        private readonly ImageUpscalerConfig _configuration;

        /// <summary>
        /// Initializes a new instance of the <see cref="UpscaleService"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="modelService">The model service.</param>
        /// <param name="imageService">The image service.</param>
        public UpscaleService(ImageUpscalerConfig configuration, IOnnxModelService modelService, IImageService imageService)
        {
            _configuration = configuration;
            _modelService = modelService;
            _imageService = imageService;
            _modelService.AddModelSet(_configuration.ModelSets);
        }


        /// <summary>
        /// Gets the configuration.
        /// </summary>
        public ImageUpscalerConfig Configuration => _configuration;


        /// <summary>
        /// Gets the model sets.
        /// </summary>
        public IReadOnlyList<UpscaleModelSet> ModelSets => _configuration.ModelSets;


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
        /// Generates an upscaled image of the source provided.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<Image<Rgba32>> GenerateAsync(UpscaleModelSet modelSet, Image<Rgba32> inputImage)
        {
            
            var upscaleInput = CreateInputParams(inputImage, modelSet.SampleSize, modelSet.ScaleFactor);
            var metadata = _modelService.GetModelMetadata(modelSet, OnnxModelType.Unet);

            var outputResult = new Image<Rgba32>(upscaleInput.OutputWidth, upscaleInput.OutputHeight);
            foreach (var tile in upscaleInput.ImageTiles)
            {
                var inputDimension = new[] { 1, modelSet.Channels, tile.Image.Height, tile.Image.Width };
                var outputDimension = new[] { 1, modelSet.Channels, tile.Destination.Height, tile.Destination.Width };
                var inputTensor = tile.Image.ToDenseTensor(inputDimension);

                using (var inferenceParameters = new OnnxInferenceParameters(metadata))
                {
                    inferenceParameters.AddInputTensor(inputTensor);
                    inferenceParameters.AddOutputBuffer(outputDimension);

                    var results = await _modelService.RunInferenceAsync(modelSet, OnnxModelType.Unet, inferenceParameters);
                    using (var result = results.First())
                    {
                        outputResult.Mutate(x => x.DrawImage(result.ToImage(), tile.Destination.Location, 1f));
                    }
                }
            }
            return outputResult;
        }


        /// <summary>
        /// Creates the input parameters.
        /// </summary>
        /// <param name="imageSource">The image source.</param>
        /// <param name="maxTileSize">Maximum size of the tile.</param>
        /// <param name="scaleFactor">The scale factor.</param>
        /// <returns></returns>
        private UpscaleInput CreateInputParams(Image<Rgba32> imageSource, int maxTileSize, int scaleFactor)
        {
            var tiles = _imageService.GenerateTiles(imageSource, maxTileSize, scaleFactor);
            var width = imageSource.Width * scaleFactor;
            var height = imageSource.Height * scaleFactor;
            return new UpscaleInput(tiles, width, height);
        }
    }
}