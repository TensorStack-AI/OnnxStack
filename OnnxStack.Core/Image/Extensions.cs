using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace OnnxStack.Core.Image
{
    public static class Extensions
    {
        public static Image<Rgba32> ToImage(this DenseTensor<float> imageTensor)
        {
            var height = imageTensor.Dimensions[2];
            var width = imageTensor.Dimensions[3];
            var hasAlpha = imageTensor.Dimensions[1] == 4;
            var result = new Image<Rgba32>(width, height);
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    result[x, y] = new Rgba32(
                        CalculateByte(imageTensor, 0, y, x),
                        CalculateByte(imageTensor, 1, y, x),
                        CalculateByte(imageTensor, 2, y, x),
                        hasAlpha ? CalculateByte(imageTensor, 3, y, x) : byte.MaxValue
                    );
                }
            }
            return result;
        }

        /// <summary>
        /// Converts to image byte array.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns></returns>
        public static byte[] ToImageBytes(this DenseTensor<float> imageTensor)
        {
            using (var image = imageTensor.ToImage())
            using (var memoryStream = new MemoryStream())
            {
                image.SaveAsPng(memoryStream);
                return memoryStream.ToArray();
            }
        }

        /// <summary>
        /// Converts to image byte array.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns></returns>
        public static async Task<byte[]> ToImageBytesAsync(this DenseTensor<float> imageTensor)
        {
            using (var image = imageTensor.ToImage())
            using (var memoryStream = new MemoryStream())
            {
                await image.SaveAsPngAsync(memoryStream);
                return memoryStream.ToArray();
            }
        }


        private static byte CalculateByte(Tensor<float> imageTensor, int index, int y, int x)
        {
            return (byte)Math.Round(Math.Clamp(imageTensor[0, index, y, x] / 2 + 0.5, 0, 1) * 255);
        }

    }
}
