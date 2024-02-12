using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace OnnxStack.Core.Image
{
    public static class Extensions
    {

        /// <summary>
        /// Converts to image mask.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns></returns>
        public static OnnxImage ToImageMask(this DenseTensor<float> imageTensor)
        {
            var width = imageTensor.Dimensions[3];
            var height = imageTensor.Dimensions[2];
            using (var result = new Image<L8>(width, height))
            {
                for (var y = 0; y < height; y++)
                {
                    for (var x = 0; x < width; x++)
                    {
                        result[x, y] = new L8((byte)(imageTensor[0, 0, y, x] * 255.0f));
                    }
                }
                return new OnnxImage(result.CloneAs<Rgba32>());
            }
        }

    }

    public enum ImageNormalizeType
    {
        ZeroToOne = 0,
        OneToOne = 1,
    }
}
