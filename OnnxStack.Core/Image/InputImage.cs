using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.IO;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace OnnxStack.Core.Image
{
    public class InputImage
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="InputImage"/> class.
        /// </summary>
        public InputImage() { }

        /// <summary>
        /// Initializes a new instance of the <see cref="InputImage"/> class.
        /// </summary>
        /// <param name="image">The image.</param>
        public InputImage(Image<Rgba32> image) => Image = image;

        /// <summary>
        /// Initializes a new instance of the <see cref="InputImage"/> class.
        /// </summary>
        /// <param name="imageBase64">The image in base64 format.</param>
        public InputImage(string imageBase64) => ImageBase64 = imageBase64;

        /// <summary>
        /// Initializes a new instance of the <see cref="InputImage"/> class.
        /// </summary>
        /// <param name="imageBytes">The image bytes.</param>
        public InputImage(byte[] imageBytes) => ImageBytes = imageBytes;

        /// <summary>
        /// Initializes a new instance of the <see cref="InputImage"/> class.
        /// </summary>
        /// <param name="imageStream">The image stream.</param>
        public InputImage(Stream imageStream) => ImageStream = imageStream;

        /// <summary>
        /// Initializes a new instance of the <see cref="InputImage"/> class.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        public InputImage(DenseTensor<float> imageTensor) => ImageTensor = imageTensor;

        /// <summary>
        /// Gets the image.
        /// </summary>
        [JsonIgnore]
        public Image<Rgba32> Image { get; set; }


        /// <summary>
        /// Gets the image base64 string.
        /// </summary>
        public string ImageBase64 { get; set; }


        /// <summary>
        /// Gets the image bytes.
        /// </summary>
        [JsonIgnore]
        public byte[] ImageBytes { get; set; }


        /// <summary>
        /// Gets the image stream.
        /// </summary>
        [JsonIgnore]
        public Stream ImageStream { get; set; }


        /// <summary>
        /// Gets the image tensor.
        /// </summary>
        [JsonIgnore]
        public DenseTensor<float> ImageTensor { get; set; }


        /// <summary>
        /// Gets a value indicating whether this instance has image.
        /// </summary>
        /// <value>
        ///   <c>true</c> if this instance has image; otherwise, <c>false</c>.
        /// </value>
        [JsonIgnore]
        public bool HasImage => Image != null
            || !string.IsNullOrEmpty(ImageBase64)
            || ImageBytes != null
            || ImageStream != null
            || ImageTensor != null;


        /// <summary>
        /// Create an image from file
        /// </summary>
        /// <param name="filePath">The file path.</param>
        /// <returns></returns>
        public static async Task<InputImage> FromFileAsync(string filePath)
        {
            return new InputImage(await File.ReadAllBytesAsync(filePath));
        }
    }
}
