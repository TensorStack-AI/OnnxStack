using Microsoft.Extensions.Primitives;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Config;
using OnnxStack.Core.Services;
using System.IO;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

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
        /// Initializes a new instance of the <see cref="VideoInput"/> class.
        /// </summary>
        /// <param name="videoFrames">The video frames.</param>
        public VideoInput(VideoFrames videoFrames) => VideoFrames = videoFrames;


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
        /// Gets or sets the video frames.
        /// </summary>
        [JsonIgnore]
        public VideoFrames VideoFrames { get; set; }


        /// <summary>
        /// Gets a value indicating whether this instance has video.
        /// </summary>
        /// <value>
        ///   <c>true</c> if this instance has video; otherwise, <c>false</c>.
        /// </value>
        [JsonIgnore]
        public bool HasVideo => VideoBytes != null
            || VideoStream != null
            || VideoTensor != null
            || VideoFrames != null;



        /// <summary>
        /// Create a VideoInput from file
        /// </summary>
        /// <param name="videoFile">The video file.</param>
        /// <param name="targetFPS">The target FPS.</param>
        /// <param name="config">The configuration.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public static async Task<VideoInput> FromFileAsync(string videoFile, float? targetFPS = default, OnnxStackConfig config = default, CancellationToken cancellationToken = default)
        {
            var videoBytes = await File.ReadAllBytesAsync(videoFile, cancellationToken);
            var videoService = new VideoService(config ?? new OnnxStackConfig());
            var videoFrames = await videoService.CreateFramesAsync(videoBytes, targetFPS, cancellationToken);
            return new VideoInput(videoFrames);
        }


        /// <summary>
        /// Saves the video file
        /// </summary>
        /// <param name="videoTensor">The video tensor.</param>
        /// <param name="videoFile">The video file.</param>
        /// <param name="targetFPS">The target FPS.</param>
        /// <param name="config">The configuration.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public static async Task SaveFileAsync(DenseTensor<float> videoTensor, string videoFile, float targetFPS, OnnxStackConfig config = default, CancellationToken cancellationToken = default)
        {
            var videoService = new VideoService(config ?? new OnnxStackConfig());
            var videoOutput = await videoService.CreateVideoAsync(videoTensor, targetFPS, cancellationToken);
            await File.WriteAllBytesAsync(videoFile, videoOutput.Data, cancellationToken);
        }
    }
}
