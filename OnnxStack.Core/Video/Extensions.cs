using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Image;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Collections.Generic;

namespace OnnxStack.Core.Video
{
    public static class Extensions
    {
        public static IEnumerable<DenseTensor<float>> ToVideoFrames(this DenseTensor<float> videoTensor)
        {
            var count = videoTensor.Dimensions[0];
            var dimensions = videoTensor.Dimensions.ToArray();
            dimensions[0] = 1;

            var newLength = (int)videoTensor.Length / count;
            for (int i = 0; i < count; i++)
            {
                var start = i * newLength;
                yield return new DenseTensor<float>(videoTensor.Buffer.Slice(start, newLength), dimensions);
            }
        }

        public static IEnumerable<byte[]> ToVideoFramesAsBytes(this DenseTensor<float> videoTensor)
        {
            foreach (var frame in videoTensor.ToVideoFrames())
            {
                yield return new OnnxImage(frame).GetImageBytes();
            }
        }

        public static async IAsyncEnumerable<byte[]> ToVideoFramesAsBytesAsync(this DenseTensor<float> videoTensor)
        {
            foreach (var frame in videoTensor.ToVideoFrames())
            {
                yield return new OnnxImage(frame).GetImageBytes();
            }
        }

        //public static IEnumerable<Image<Rgba32>> ToVideoFramesAsImage(this DenseTensor<float> videoTensor)
        //{
        //    foreach (var frame in videoTensor.ToVideoFrames())
        //    {
        //        yield return frame.ToImage();
        //    }
        //}
    }
}
