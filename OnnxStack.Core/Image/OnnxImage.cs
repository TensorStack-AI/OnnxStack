using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.IO;
using System.Threading.Tasks;
using ImageSharp = SixLabors.ImageSharp.Image;

namespace OnnxStack.Core.Image
{
    public sealed class OnnxImage : IDisposable
    {
        private readonly Image<Rgba32> _imageData;


        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxImage"/> class.
        /// </summary>
        /// <param name="image">The image.</param>
        public OnnxImage(Image<Rgba32> image)
        {
            _imageData = image.Clone();
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxImage"/> class.
        /// </summary>
        /// <param name="filename">The filename.</param>
        public OnnxImage(string filename)
        {
            _imageData = ImageSharp.Load<Rgba32>(filename);
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxImage"/> class.
        /// </summary>
        /// <param name="imageBytes">The image bytes.</param>
        public OnnxImage(byte[] imageBytes)
        {
            _imageData = ImageSharp.Load<Rgba32>(imageBytes);
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxImage"/> class.
        /// </summary>
        /// <param name="imageStream">The image stream.</param>
        public OnnxImage(Stream imageStream)
        {
            _imageData = ImageSharp.Load<Rgba32>(imageStream);
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxImage"/> class.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="normalizeType">Type of the normalize.</param>
        public OnnxImage(DenseTensor<float> imageTensor, ImageNormalizeType normalizeType = ImageNormalizeType.OneToOne)
        {
            var height = imageTensor.Dimensions[2];
            var width = imageTensor.Dimensions[3];
            var hasTransparency = imageTensor.Dimensions[1] == 4;
            _imageData = new Image<Rgba32>(width, height);
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    if (normalizeType == ImageNormalizeType.ZeroToOne)
                    {
                        _imageData[x, y] = new Rgba32(
                            DenormalizeZeroToOneToByte(imageTensor, 0, y, x),
                            DenormalizeZeroToOneToByte(imageTensor, 1, y, x),
                            DenormalizeZeroToOneToByte(imageTensor, 2, y, x),
                            hasTransparency ? DenormalizeZeroToOneToByte(imageTensor, 3, y, x) : byte.MaxValue);
                    }
                    else
                    {
                        _imageData[x, y] = new Rgba32(
                            DenormalizeOneToOneToByte(imageTensor, 0, y, x),
                            DenormalizeOneToOneToByte(imageTensor, 1, y, x),
                            DenormalizeOneToOneToByte(imageTensor, 2, y, x),
                            hasTransparency ? DenormalizeOneToOneToByte(imageTensor, 3, y, x) : byte.MaxValue);
                    }
                }
            }
        }


        /// <summary>
        /// Gets the height.
        /// </summary>
        public int Height => _imageData.Height;

        /// <summary>
        /// Gets the width.
        /// </summary>
        public int Width => _imageData.Width;

        /// <summary>
        /// Gets a value indicating whether this instance has image.
        /// </summary>
        /// <value>
        ///   <c>true</c> if this instance has image; otherwise, <c>false</c>.
        /// </value>
        public bool HasImage
        {
            get { return _imageData != null; }
        }


        /// <summary>
        /// Gets the image.
        /// </summary>
        /// <returns></returns>
        public Image<Rgba32> GetImage()
        {
            return _imageData;
        }


        /// <summary>
        /// Gets the image as base64.
        /// </summary>
        /// <returns></returns>
        public string GetImageBase64()
        {
            return _imageData?.ToBase64String(PngFormat.Instance);
        }


        /// <summary>
        /// Gets the image as bytes.
        /// </summary>
        /// <returns></returns>
        public byte[] GetImageBytes()
        {
            using (var memoryStream = new MemoryStream())
            {
                _imageData.SaveAsPng(memoryStream);
                return memoryStream.ToArray();
            }
        }


        /// <summary>
        /// Gets the image as bytes.
        /// </summary>
        /// <returns></returns>
        public async Task<byte[]> GetImageBytesAsync()
        {
            using (var memoryStream = new MemoryStream())
            {
                await _imageData.SaveAsPngAsync(memoryStream);
                return memoryStream.ToArray();
            }
        }


        /// <summary>
        /// Gets the image as stream.
        /// </summary>
        /// <returns></returns>
        public Stream GetImageStream()
        {
            var memoryStream = new MemoryStream();
            _imageData.SaveAsPng(memoryStream);
            return memoryStream;
        }


        /// <summary>
        /// Gets the image as stream.
        /// </summary>
        /// <returns></returns>
        public async Task<Stream> GetImageStreamAsync()
        {
            var memoryStream = new MemoryStream();
            await _imageData.SaveAsPngAsync(memoryStream);
            return memoryStream;
        }


        /// <summary>
        /// Copies the image to stream.
        /// </summary>
        /// <param name="destination">The destination.</param>
        /// <returns></returns>
        public void CopyToStream(Stream destination)
        {
            _imageData.SaveAsPng(destination);
        }


        /// <summary>
        /// Copies the image to stream.
        /// </summary>
        /// <param name="destination">The destination.</param>
        /// <returns></returns>
        public Task CopyToStreamAsync(Stream destination)
        {
            return _imageData.SaveAsPngAsync(destination);
        }


        /// <summary>
        /// Gets the image as tensor.
        /// </summary>
        /// <param name="normalizeType">Type of the normalize.</param>
        /// <param name="channels">The channels.</param>
        /// <returns></returns>
        public DenseTensor<float> GetImageTensor(ImageNormalizeType normalizeType = ImageNormalizeType.OneToOne, int channels = 3)
        {
            var dimensions = new[] { 1, channels, Height, Width };
            return normalizeType == ImageNormalizeType.ZeroToOne
                ? NormalizeToZeroToOne(dimensions)
                : NormalizeToOneToOne(dimensions);
        }


        /// <summary>
        /// Gets the image as tensor.
        /// </summary>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <param name="normalizeType">Type of the normalize.</param>
        /// <param name="channels">The channels.</param>
        /// <returns></returns>
        public DenseTensor<float> GetImageTensor(int height, int width, ImageNormalizeType normalizeType = ImageNormalizeType.OneToOne, int channels = 3, ImageResizeMode resizeMode = ImageResizeMode.Crop)
        {
            if (height > 0 && width > 0)
                Resize(height, width, resizeMode);

            return GetImageTensor(normalizeType, channels);
        }


        /// <summary>
        /// Gets the image as tensor asynchronously.
        /// </summary>
        /// <param name="normalizeType">Type of the normalize.</param>
        /// <param name="channels">The channels.</param>
        /// <returns></returns>
        public Task<DenseTensor<float>> GetImageTensorAsync(ImageNormalizeType normalizeType = ImageNormalizeType.OneToOne, int channels = 3)
        {
            return Task.Run(() => GetImageTensor(normalizeType, channels));
        }


        /// <summary>
        /// Gets the image as tensor asynchronously.
        /// </summary>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <param name="normalizeType">Type of the normalize.</param>
        /// <param name="channels">The channels.</param>
        /// <returns></returns>
        public Task<DenseTensor<float>> GetImageTensorAsync(int height, int width, ImageNormalizeType normalizeType = ImageNormalizeType.OneToOne, int channels = 3, ImageResizeMode resizeMode = ImageResizeMode.Crop)
        {
            return Task.Run(() => GetImageTensor(height, width, normalizeType, channels, resizeMode));
        }


        /// <summary>
        /// Resizes the image.
        /// </summary>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <param name="resizeMode">The resize mode.</param>
        public void Resize(int height, int width, ImageResizeMode resizeMode = ImageResizeMode.Crop)
        {
            _imageData.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(width, height),
                    Mode = resizeMode.ToResizeMode(),
                    Sampler = KnownResamplers.Lanczos8,
                    Compand = true
                });
            });
        }

  
        public OnnxImage Clone()
        {
            return new OnnxImage(_imageData);
        }

        /// <summary>
        /// Saves the specified image to file.
        /// </summary>
        /// <param name="filename">The filename.</param>
        public void Save(string filename)
        {
            _imageData?.SaveAsPng(filename);
        }


        /// <summary>
        /// Saves the specified image to file asynchronously.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <returns></returns>
        public Task SaveAsync(string filename)
        {
            return _imageData?.SaveAsPngAsync(filename);
        }


        /// <summary>
        /// Saves the specified image to stream.
        /// </summary>
        /// <param name="stream">The stream.</param>
        public void Save(Stream stream)
        {
            _imageData?.SaveAsPng(stream);
        }


        /// <summary>
        /// Saves the specified image to stream asynchronously.
        /// </summary>
        /// <param name="stream">The stream.</param>
        /// <returns></returns>
        public Task SaveAsync(Stream stream)
        {
            return _imageData?.SaveAsPngAsync(stream);
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            _imageData?.Dispose();
        }


        /// <summary>
        /// Normalizes the pixels from 0-255 to 0-1
        /// </summary>
        /// <param name="image">The image.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        private DenseTensor<float> NormalizeToZeroToOne(ReadOnlySpan<int> dimensions)
        {
            var width = dimensions[3];
            var height = dimensions[2];
            var channels = dimensions[1];
            var hasTransparency = channels == 4;
            var imageArray = new DenseTensor<float>(new[] { 1, channels, height, width });
            _imageData.ProcessPixelRows(img =>
            {
                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        var pixelSpan = img.GetRowSpan(y);
                        imageArray[0, 0, y, x] = (pixelSpan[x].R / 255.0f);
                        imageArray[0, 1, y, x] = (pixelSpan[x].G / 255.0f);
                        imageArray[0, 2, y, x] = (pixelSpan[x].B / 255.0f);
                        if (hasTransparency)
                            imageArray[0, 3, y, x] = (pixelSpan[x].A / 255.0f);
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
        private DenseTensor<float> NormalizeToOneToOne(ReadOnlySpan<int> dimensions)
        {
            var width = dimensions[3];
            var height = dimensions[2];
            var channels = dimensions[1];
            var hasTransparency = channels == 4;
            var imageArray = new DenseTensor<float>(new[] { 1, channels, height, width });
            _imageData.ProcessPixelRows(img =>
            {
                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        var pixelSpan = img.GetRowSpan(y);
                        imageArray[0, 0, y, x] = (pixelSpan[x].R / 255.0f) * 2.0f - 1.0f;
                        imageArray[0, 1, y, x] = (pixelSpan[x].G / 255.0f) * 2.0f - 1.0f;
                        imageArray[0, 2, y, x] = (pixelSpan[x].B / 255.0f) * 2.0f - 1.0f;
                        if (hasTransparency)
                            imageArray[0, 3, y, x] = (pixelSpan[x].A / 255.0f) * 2.0f - 1.0f;
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


        /// <summary>
        /// Create OnnxImage from file asynchronously
        /// </summary>
        /// <param name="filePath">The file path.</param>
        /// <returns></returns>
        public static async Task<OnnxImage> FromFileAsync(string filePath)
        {
            return new OnnxImage(await ImageSharp.LoadAsync<Rgba32>(filePath));
        }


        /// <summary>
        /// Create OnnxImage from stream asynchronously
        /// </summary>
        /// <param name="imageStream">The image stream.</param>
        /// <returns></returns>
        public static async Task<OnnxImage> FromStreamAsync(Stream imageStream)
        {
            return new OnnxImage(await ImageSharp.LoadAsync<Rgba32>(imageStream));
        }


        /// <summary>
        /// Create OnnxImage from bytes asynchronously
        /// </summary>
        /// <param name="imageStream">The image stream.</param>
        /// <returns></returns>
        public static async Task<OnnxImage> FromBytesAsync(byte[] imageBytes)
        {
            return await Task.Run(() => new OnnxImage(imageBytes));
        }
    }
}
