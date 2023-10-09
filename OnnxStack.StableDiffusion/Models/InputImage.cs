using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.IO;
using System.Text.Json.Serialization;

namespace OnnxStack.StableDiffusion.Models
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
        public InputImage(Image<Rgb24> image) => Image = image;

        /// <summary>
        /// Initializes a new instance of the <see cref="InputImage"/> class.
        /// </summary>
        /// <param name="imagePath">The image path.</param>
        public InputImage(string imagePath) => ImagePath = imagePath;

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
        public Image<Rgb24> Image { get; set; }


        /// <summary>
        /// Gets the image path.
        /// </summary>
        public string ImagePath { get; set; }


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
            || !string.IsNullOrEmpty(ImagePath)
            || ImageBytes != null
            || ImageStream != null
            || ImageTensor != null;
    }
}
