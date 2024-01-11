using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.IO;
using System.Threading.Tasks;
using ImageSharp = SixLabors.ImageSharp.Image;

namespace OnnxStack.Core.Image
{
    public static class Extensions
    {
        #region Image To Image

        /// <summary>
        /// Converts to InputImage to an Image<Rgba32>
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public static Image<Rgba32> ToImage(this InputImage inputImage)
        {
            if (!string.IsNullOrEmpty(inputImage.ImageBase64))
                return ImageSharp.Load<Rgba32>(Convert.FromBase64String(inputImage.ImageBase64.Split(',')[1]));
            if (inputImage.ImageBytes != null)
                return ImageSharp.Load<Rgba32>(inputImage.ImageBytes);
            if (inputImage.ImageStream != null)
                return ImageSharp.Load<Rgba32>(inputImage.ImageStream);
            if (inputImage.ImageTensor != null)
                return inputImage.ImageTensor.ToImage();

            return inputImage.Image;
        }


        /// <summary>
        /// Converts to InputImage to an Image<Rgba32>
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public static async Task<Image<Rgba32>> ToImageAsync(this InputImage inputImage)
        {
            return await Task.Run(inputImage.ToImage);
        }


        /// <summary>
        /// Resizes the specified image.
        /// </summary>
        /// <param name="image">The image.</param>
        /// <param name="dimensions">The dimensions.</param>
        public static void Resize(this Image<Rgba32> image, int height, int width, ResizeMode resizeMode = ResizeMode.Crop)
        {
            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(width, height),
                    Mode = resizeMode,
                    Sampler = KnownResamplers.Lanczos8,
                    Compand = true
                });
            });
        }

        #endregion

        #region Tensor To Image

        /// <summary>
        /// Converts to image byte array.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns></returns>
        public static byte[] ToImageBytes(this DenseTensor<float> imageTensor, ImageNormalizeType imageNormalizeType = ImageNormalizeType.OneToOne)
        {
            using (var image = imageTensor.ToImage(imageNormalizeType))
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
        public static async Task<byte[]> ToImageBytesAsync(this DenseTensor<float> imageTensor, ImageNormalizeType imageNormalizeType = ImageNormalizeType.OneToOne)
        {
            using (var image = imageTensor.ToImage(imageNormalizeType))
            using (var memoryStream = new MemoryStream())
            {
                await image.SaveAsPngAsync(memoryStream);
                return memoryStream.ToArray();
            }
        }


        /// <summary>
        /// Converts to image memory stream.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns></returns>
        public static Stream ToImageStream(this DenseTensor<float> imageTensor, ImageNormalizeType imageNormalizeType = ImageNormalizeType.OneToOne)
        {
            using (var image = imageTensor.ToImage(imageNormalizeType))
            {
                var memoryStream = new MemoryStream();
                image.SaveAsPng(memoryStream);
                return memoryStream;
            }
        }


        /// <summary>
        /// Converts to image memory stream.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns></returns>
        public static async Task<Stream> ToImageStreamAsync(this DenseTensor<float> imageTensor, ImageNormalizeType imageNormalizeType = ImageNormalizeType.OneToOne)
        {
            using (var image = imageTensor.ToImage(imageNormalizeType))
            {
                var memoryStream = new MemoryStream();
                await image.SaveAsPngAsync(memoryStream);
                return memoryStream;
            }
        }


        /// <summary>
        /// Converts to image mask.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns></returns>
        public static Image<Rgba32> ToImageMask(this DenseTensor<float> imageTensor)
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
                return result.CloneAs<Rgba32>();
            }
        }


        /// <summary>
        /// Converts to Image.
        /// </summary>
        /// <param name="ortValue">The ort value.</param>
        /// <returns></returns>
        public static Image<Rgba32> ToImage(this OrtValue ortValue, ImageNormalizeType normalizeType = ImageNormalizeType.OneToOne)
        {
            return ortValue.ToDenseTensor().ToImage(normalizeType);
        }


        /// <summary>
        /// Converts to image.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns></returns>
        public static Image<Rgba32> ToImage(this DenseTensor<float> imageTensor, ImageNormalizeType normalizeType = ImageNormalizeType.OneToOne)
        {
            var height = imageTensor.Dimensions[2];
            var width = imageTensor.Dimensions[3];
            var result = new Image<Rgba32>(width, height);
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    if (normalizeType == ImageNormalizeType.ZeroToOne)
                    {
                        result[x, y] = new Rgba32(
                            DenormalizeZeroToOneToByte(imageTensor, 0, y, x),
                            DenormalizeZeroToOneToByte(imageTensor, 1, y, x),
                            DenormalizeZeroToOneToByte(imageTensor, 2, y, x));
                    }
                    else
                    {
                        result[x, y] = new Rgba32(
                            DenormalizeOneToOneToByte(imageTensor, 0, y, x),
                            DenormalizeOneToOneToByte(imageTensor, 1, y, x),
                            DenormalizeOneToOneToByte(imageTensor, 2, y, x));
                    }
                }
            }
            return result;
        }

        #endregion

        #region Image To Tensor


        /// <summary>
        /// Converts to DenseTensor.
        /// </summary>
        /// <param name="image">The image.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        public static DenseTensor<float> ToDenseTensor(this Image<Rgba32> image, ImageNormalizeType normalizeType = ImageNormalizeType.OneToOne, int channels = 3)
        {
            var dimensions = new[] { 1, channels, image.Height, image.Width };
            return normalizeType == ImageNormalizeType.ZeroToOne
                ? NormalizeToZeroToOne(image, dimensions)
                : NormalizeToOneToOne(image, dimensions);
        }


        /// <summary>
        /// Converts to InputImage to DenseTensor<float>.
        /// </summary>
        /// <param name="imageData">The image data.</param>
        /// <param name="imageNormalizeType">Type of the image normalize.</param>
        /// <returns></returns>
        public static async Task<DenseTensor<float>> ToDenseTensorAsync(this InputImage imageData, ImageNormalizeType imageNormalizeType = ImageNormalizeType.OneToOne)
        {
            if (!string.IsNullOrEmpty(imageData.ImageBase64))
                return await TensorFromBase64Async(imageData.ImageBase64, default, default, imageNormalizeType);
            if (imageData.ImageBytes != null)
                return await TensorFromBytesAsync(imageData.ImageBytes, default, default, imageNormalizeType);
            if (imageData.ImageStream != null)
                return await TensorFromStreamAsync(imageData.ImageStream, default, default, imageNormalizeType);
            if (imageData.ImageTensor != null)
                return imageData.ImageTensor.ToDenseTensor(); // Note: Tensor Copy // TODO: Reshape to dimensions

            return await TensorFromImageAsync(imageData.Image, default, default, imageNormalizeType);
        }


        /// <summary>
        /// Converts to InputImage to DenseTensor<float>.
        /// </summary>
        /// <param name="imageData">The image data.</param>
        /// <param name="resizeHeight">Height of the resize.</param>
        /// <param name="resizeWidth">Width of the resize.</param>
        /// <param name="imageNormalizeType">Type of the image normalize.</param>
        /// <returns></returns>
        public static async Task<DenseTensor<float>> ToDenseTensorAsync(this InputImage imageData, int resizeHeight, int resizeWidth, ImageNormalizeType imageNormalizeType = ImageNormalizeType.OneToOne)
        {
            if (!string.IsNullOrEmpty(imageData.ImageBase64))
                return await TensorFromBase64Async(imageData.ImageBase64, resizeHeight, resizeWidth, imageNormalizeType);
            if (imageData.ImageBytes != null)
                return await TensorFromBytesAsync(imageData.ImageBytes, resizeHeight, resizeWidth, imageNormalizeType);
            if (imageData.ImageStream != null)
                return await TensorFromStreamAsync(imageData.ImageStream, resizeHeight, resizeWidth, imageNormalizeType);
            if (imageData.ImageTensor != null)
                return imageData.ImageTensor.ToDenseTensor(); // Note: Tensor Copy // TODO: Reshape to dimensions

            return await TensorFromImageAsync(imageData.Image, resizeHeight, resizeWidth, imageNormalizeType);
        }


        /// <summary>
        /// Tensor from image.
        /// </summary>
        /// <param name="image">The image.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <param name="imageNormalizeType">Type of the image normalize.</param>
        /// <returns></returns>
        private static DenseTensor<float> TensorFromImage(Image<Rgba32> image, int height, int width, ImageNormalizeType imageNormalizeType)
        {
            if (height > 0 && width > 0)
                image.Resize(height, width);

            return image.ToDenseTensor(imageNormalizeType);
        }


        /// <summary>
        /// Tensor from image.
        /// </summary>
        /// <param name="image">The image.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <param name="imageNormalizeType">Type of the image normalize.</param>
        /// <returns></returns>
        private static Task<DenseTensor<float>> TensorFromImageAsync(Image<Rgba32> image, int height, int width, ImageNormalizeType imageNormalizeType)
        {
            return Task.Run(() => TensorFromImage(image, height, width, imageNormalizeType));
        }


        /// <summary>
        /// Tensor from image file.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <param name="imageNormalizeType">Type of the image normalize.</param>
        /// <returns></returns>
        private static DenseTensor<float> TensorFromFile(string filename, int height, int width, ImageNormalizeType imageNormalizeType)
        {
            using (var image = ImageSharp.Load<Rgba32>(filename))
            {
                if (height > 0 && width > 0)
                    image.Resize(height, width);

                return image.ToDenseTensor(imageNormalizeType);
            }
        }


        /// <summary>
        /// Tensor from image file.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <param name="imageNormalizeType">Type of the image normalize.</param>
        /// <returns></returns>
        private static async Task<DenseTensor<float>> TensorFromFileAsync(string filename, int height, int width, ImageNormalizeType imageNormalizeType)
        {
            using (var image = await ImageSharp.LoadAsync<Rgba32>(filename))
            {
                if (height > 0 && width > 0)
                    image.Resize(height, width);

                return image.ToDenseTensor(imageNormalizeType);
            }
        }


        /// <summary>
        /// Tensor from base64 image.
        /// </summary>
        /// <param name="base64Image">The base64 image.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <param name="imageNormalizeType">Type of the image normalize.</param>
        /// <returns></returns>
        private static DenseTensor<float> TensorFromBase64(string base64Image, int height, int width, ImageNormalizeType imageNormalizeType)
        {
            return TensorFromBytes(Convert.FromBase64String(base64Image.Split(',')[1]), height, width, imageNormalizeType);
        }


        /// <summary>
        /// Tensor from base64 image.
        /// </summary>
        /// <param name="base64Image">The base64 image.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <param name="imageNormalizeType">Type of the image normalize.</param>
        /// <returns></returns>
        private static async Task<DenseTensor<float>> TensorFromBase64Async(string base64Image, int height, int width, ImageNormalizeType imageNormalizeType)
        {
            return await TensorFromBytesAsync(Convert.FromBase64String(base64Image.Split(',')[1]), height, width, imageNormalizeType);
        }


        /// <summary>
        /// Tensor from image bytes.
        /// </summary>
        /// <param name="imageBytes">The image bytes.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <param name="imageNormalizeType">Type of the image normalize.</param>
        /// <returns></returns>
        private static DenseTensor<float> TensorFromBytes(byte[] imageBytes, int height, int width, ImageNormalizeType imageNormalizeType)
        {
            using (var image = ImageSharp.Load<Rgba32>(imageBytes))
            {
                if (height > 0 && width > 0)
                    image.Resize(height, width);

                return image.ToDenseTensor(imageNormalizeType);
            }
        }


        /// <summary>
        /// Tensors from image bytes.
        /// </summary>
        /// <param name="imageBytes">The image bytes.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <param name="imageNormalizeType">Type of the image normalize.</param>
        /// <returns></returns>
        private static async Task<DenseTensor<float>> TensorFromBytesAsync(byte[] imageBytes, int height, int width, ImageNormalizeType imageNormalizeType)
        {
            return await Task.Run(() => TensorFromBytes(imageBytes, height, width, imageNormalizeType));
        }


        /// <summary>
        /// Tensor from image stream.
        /// </summary>
        /// <param name="imageStream">The image stream.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <param name="imageNormalizeType">Type of the image normalize.</param>
        /// <returns></returns>
        private static DenseTensor<float> TensorFromStream(Stream imageStream, int height, int width, ImageNormalizeType imageNormalizeType)
        {
            using (var image = ImageSharp.Load<Rgba32>(imageStream))
            {
                if (height > 0 && width > 0)
                    image.Resize(height, width);

                return image.ToDenseTensor(imageNormalizeType);
            }
        }


        /// <summary>
        /// Tensor from image stream.
        /// </summary>
        /// <param name="imageStream">The image stream.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <param name="imageNormalizeType">Type of the image normalize.</param>
        /// <returns></returns>
        private static async Task<DenseTensor<float>> TensorFromStreamAsync(Stream imageStream, int height, int width, ImageNormalizeType imageNormalizeType)
        {
            using (var image = await ImageSharp.LoadAsync<Rgba32>(imageStream))
            {
                if (height > 0 && width > 0)
                    image.Resize(height, width);

                return image.ToDenseTensor(imageNormalizeType);
            }
        }

        #endregion

        #region Normalize

        /// <summary>
        /// Normalizes the pixels from 0-255 to 0-1
        /// </summary>
        /// <param name="image">The image.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        private static DenseTensor<float> NormalizeToZeroToOne(Image<Rgba32> image, ReadOnlySpan<int> dimensions)
        {
            var width = dimensions[3];
            var height = dimensions[2];
            var channels = dimensions[1];
            var imageArray = new DenseTensor<float>(new[] { 1, channels, height, width });
            image.ProcessPixelRows(img =>
            {
                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        var pixelSpan = img.GetRowSpan(y);
                        imageArray[0, 0, y, x] = (pixelSpan[x].R / 255.0f);
                        imageArray[0, 1, y, x] = (pixelSpan[x].G / 255.0f);
                        imageArray[0, 2, y, x] = (pixelSpan[x].B / 255.0f);
                    }
                }
            });
            return imageArray;
        }


        /// <summary>
        /// Normalizes the pixels from 0-255 to 0-1
        /// </summary>
        /// <param name="image">The image.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        private static DenseTensor<float> NormalizeToOneToOne(Image<Rgba32> image, ReadOnlySpan<int> dimensions)
        {
            var width = dimensions[3];
            var height = dimensions[2];
            var channels = dimensions[1];
            var imageArray = new DenseTensor<float>(new[] { 1, channels, height, width });
            image.ProcessPixelRows(img =>
            {
                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        var pixelSpan = img.GetRowSpan(y);
                        imageArray[0, 0, y, x] = (pixelSpan[x].R / 255.0f) * 2.0f - 1.0f;
                        imageArray[0, 1, y, x] = (pixelSpan[x].G / 255.0f) * 2.0f - 1.0f;
                        imageArray[0, 2, y, x] = (pixelSpan[x].B / 255.0f) * 2.0f - 1.0f;
                    }
                }
            });
            return imageArray;
        }


        /// <summary>
        /// Denormalizes the pixels from 0 to 1 to 0-255
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="index">The index.</param>
        /// <param name="y">The y.</param>
        /// <param name="x">The x.</param>
        /// <returns></returns>
        private static byte DenormalizeZeroToOneToByte(DenseTensor<float> imageTensor, int index, int y, int x)
        {
            return (byte)Math.Round(Math.Clamp(imageTensor[0, index, y, x], 0, 1) * 255);
        }


        /// <summary>
        /// Denormalizes the pixels from -1 to 1 to 0-255
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="index">The index.</param>
        /// <param name="y">The y.</param>
        /// <param name="x">The x.</param>
        /// <returns></returns>
        private static byte DenormalizeOneToOneToByte(Tensor<float> imageTensor, int index, int y, int x)
        {
            return (byte)Math.Round(Math.Clamp(imageTensor[0, index, y, x] / 2 + 0.5, 0, 1) * 255);
        }

        #endregion
    }

    public enum ImageNormalizeType
    {
        ZeroToOne = 0,
        OneToOne = 1,
    }
}
