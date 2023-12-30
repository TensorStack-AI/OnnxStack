using Microsoft.ML.OnnxRuntime.Tensors;
using System.IO;
using System.Text.Json.Serialization;

namespace OnnxStack.Core.Video
{
    public class VideoInput
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="VideoInput"/> class.
        /// </summary>
        public VideoInput() { }

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoInput"/> class.
        /// </summary>
        /// <param name="videoBytes">The video bytes.</param>
        public VideoInput(byte[] videoBytes) => VideoBytes = videoBytes;

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoInput"/> class.
        /// </summary>
        /// <param name="videoStream">The video stream.</param>
        public VideoInput(Stream videoStream) => VideoStream = videoStream;

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoInput"/> class.
        /// </summary>
        /// <param name="videoTensor">The video tensor.</param>
        public VideoInput(DenseTensor<float> videoTensor) => VideoTensor = videoTensor;


        /// <summary>
        /// Gets the video bytes.
        /// </summary>
        [JsonIgnore]
        public byte[] VideoBytes { get; set; }


        /// <summary>
        /// Gets the video stream.
        /// </summary>
        [JsonIgnore]
        public Stream VideoStream { get; set; }


        /// <summary>
        /// Gets the video tensor.
        /// </summary>
        [JsonIgnore]
        public DenseTensor<float> VideoTensor { get; set; }


        /// <summary>
        /// Gets a value indicating whether this instance has video.
        /// </summary>
        /// <value>
        ///   <c>true</c> if this instance has video; otherwise, <c>false</c>.
        /// </value>
        [JsonIgnore]
        public bool HasVideo => VideoBytes != null
            || VideoStream != null
            || VideoTensor != null;
    }
}
