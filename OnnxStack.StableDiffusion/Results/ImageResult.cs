using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using System;
using OnnxStack.StableDiffusion.Config;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Results
{
    public class ImageResult
    {
        public ImageResult(Image<Rgba32> image)
        {
            Image = image;
        }

        public ImageResult(StableDiffusionOptions options, Image<Rgba32> image)
        {
            Image = image;
            Seed = options.Seed;
            Steps = options.NumInferenceSteps;
            Guidance = options.GuidanceScale;
        }

        public Image<Rgba32> Image { get; }
        public int Seed { get; }
        public int Steps { get; }
        public double Guidance { get; }
        public string FileName { get; set; }


        /// <summary>
        /// Saves the file.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <exception cref="System.ArgumentNullException"></exception>
        public async Task SaveAsync(string filename)
        {
            ArgumentNullException.ThrowIfNull(Image);
            ArgumentException.ThrowIfNullOrEmpty(filename);

            FileName = filename;
            await Image.SaveAsync(filename).ConfigureAwait(false);
        }
    }

}
