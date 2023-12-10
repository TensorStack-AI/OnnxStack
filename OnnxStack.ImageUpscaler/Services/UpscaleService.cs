using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.Core.Services;
using OnnxStack.ImageUpscaler.Config;
using OnnxStack.ImageUpscaler.Extensions;
using OnnxStack.ImageUpscaler.Models;
using OnnxStack.StableDiffusion.Config;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace OnnxStack.ImageUpscaler.Services
{
    public class UpscaleService : IUpscaleService
    {
        private readonly IOnnxModelService _modelService;
        private readonly ImageUpscalerConfig _configuration;
        private readonly HashSet<UpscaleModelSet> _modelSetConfigs;

        /// <summary>
        /// Initializes a new instance of the <see cref="UpscaleService"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="modelService">The model service.</param>
        /// <param name="imageService">The image service.</param>
        public UpscaleService(ImageUpscalerConfig configuration, IOnnxModelService modelService)
        {
            _configuration = configuration;
            _modelService = modelService;
            _modelSetConfigs = new HashSet<UpscaleModelSet>(_configuration.ModelSets, new OnnxModelEqualityComparer());
            _modelService.AddModelSet(_modelSetConfigs);
        }


        /// <summary>
        /// Gets the model sets.
        /// </summary>
        public IReadOnlyCollection<UpscaleModelSet> ModelSets => _modelSetConfigs;


        /// <summary>
        /// Adds the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        /// <exception cref="System.NotImplementedException"></exception>
        public async Task<bool> AddModelAsync(UpscaleModelSet model)
        {
            if (await _modelService.AddModelSet(model))
            {
                _modelSetConfigs.Add(model);
                return true;
            }
            return false;
        }


        /// <summary>
        /// Removes the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        /// <exception cref="System.NotImplementedException"></exception>
        public async Task<bool> RemoveModelAsync(UpscaleModelSet model)
        {
            if (await _modelService.RemoveModelSet(model))
            {
                _modelSetConfigs.Remove(model);
                return true;
            }
            return false;
        }


        /// <summary>
        /// Updates the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        /// <exception cref="System.NotImplementedException"></exception>
        public async Task<bool> UpdateModelAsync(UpscaleModelSet model)
        {
            if (await _modelService.UpdateModelSet(model))
            {
                _modelSetConfigs.Remove(model);
                _modelSetConfigs.Add(model);
                return true;
            }
            return false;
        }


        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        public async Task<bool> LoadModelAsync(UpscaleModelSet model)
        {
            if (!_modelSetConfigs.TryGetValue(model, out _))
                throw new Exception("ModelSet not found");

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
        /// Generates the upscaled image.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<DenseTensor<float>> GenerateAsync(UpscaleModelSet modelOptions, InputImage inputImage)
        {
            var image = await GenerateInternalAsync(modelOptions, inputImage);
            return image.ToDenseTensor(new[] { 1, modelOptions.Channels, image.Height, image.Width });
        }


        /// <summary>
        /// Generates the upscaled image.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<Image<Rgba32>> GenerateAsImageAsync(UpscaleModelSet modelOptions, InputImage inputImage)
        {
            return await GenerateInternalAsync(modelOptions, inputImage);
        }


        /// <summary>
        /// Generates the upscaled image.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<byte[]> GenerateAsByteAsync(UpscaleModelSet modelOptions, InputImage inputImage)
        {
            using (var memoryStream = new MemoryStream())
            {
                var image = await GenerateInternalAsync(modelOptions, inputImage);
                image.SaveAsPng(memoryStream);
                return memoryStream.ToArray();
            }
        }


        /// <summary>
        /// Generates the upscaled image.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<Stream> GenerateAsStreamAsync(UpscaleModelSet modelOptions, InputImage inputImage)
        {
            var image = await GenerateInternalAsync(modelOptions, inputImage);
            var memoryStream = new MemoryStream();
            image.SaveAsPng(memoryStream);
            return memoryStream;
        }


        /// <summary>
        /// Generates an upscaled image of the source provided.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="inputImage">The input image.</param>
        private async Task<Image<Rgba32>> GenerateInternalAsync(UpscaleModelSet modelSet, InputImage inputImage)
        {
            using (var image = inputImage.ToImage())
            {
                var upscaleInput = CreateInputParams(image, modelSet.SampleSize, modelSet.ScaleFactor);
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
            var tiles = imageSource.GenerateTiles(maxTileSize, scaleFactor);
            var width = imageSource.Width * scaleFactor;
            var height = imageSource.Height * scaleFactor;
            return new UpscaleInput(tiles, width, height);
        }
    }
}