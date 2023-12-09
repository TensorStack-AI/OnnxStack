using OnnxStack.ImageUpscaler.Config;
using OnnxStack.StableDiffusion.Config;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace OnnxStack.ImageUpscaler.Services
{
    public interface IUpscaleService
    {

        /// <summary>
        /// Gets the configuration.
        /// </summary>
        ImageUpscalerConfig Configuration { get; }


        /// <summary>
        /// Gets the model sets.
        /// </summary>
        IReadOnlyList<UpscaleModelSet> ModelSets { get; }


        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        Task<bool> LoadModelAsync(UpscaleModelSet model);


        /// <summary>
        /// Unloads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        Task<bool> UnloadModelAsync(UpscaleModelSet model);


        /// <summary>
        /// Generates the upscaled image.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        Task<Image<Rgba32>> GenerateAsync(UpscaleModelSet modelOptions, Image<Rgba32> inputImage);
    }
}
