using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using OnnxStack.Core.Services;
using OnnxStack.ImageUpscaler.Extensions;
using OnnxStack.ImageUpscaler.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Linq;
using System.Threading.Tasks;

namespace OnnxStack.ImageUpscaler.Services
{
    public class UpscaleService : IUpscaleService
    {
        private readonly IImageService _imageService;
        private readonly IOnnxModelService _modelService;

        public UpscaleService(IOnnxModelService modelService, IImageService imageService)
        {
            _modelService = modelService;
            _imageService = imageService;
        }


        /// <summary>
        /// Generates an upscaled image of the source provided.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<Image<Rgba32>> GenerateAsync(IOnnxModel modelOptions, Image<Rgba32> inputImage)
        {
            var channels = 3;
            var maxSize = 512;
            var scaleFactor = 4;
            var upscaleInput = CreateInputParams(inputImage, maxSize, scaleFactor);

            await _modelService.LoadModelAsync(modelOptions);
            var metadata = _modelService.GetModelMetadata(modelOptions, OnnxModelType.Upscaler);

            var outputResult = new Image<Rgba32>(upscaleInput.OutputWidth, upscaleInput.OutputHeight);
            foreach (var tile in upscaleInput.ImageTiles)
            {
                var inputDimension = new[] { 1, channels, tile.Image.Height, tile.Image.Width };
                var outputDimension = new[] { 1, channels, tile.Destination.Height, tile.Destination.Width };
                var inputTensor = tile.Image.ToDenseTensor(inputDimension);

                using (var inferenceParameters = new OnnxInferenceParameters(metadata))
                {
                    inferenceParameters.AddInputTensor(inputTensor);
                    inferenceParameters.AddOutputBuffer(outputDimension);

                    var results = await _modelService.RunInferenceAsync(modelOptions, OnnxModelType.Upscaler, inferenceParameters);
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



