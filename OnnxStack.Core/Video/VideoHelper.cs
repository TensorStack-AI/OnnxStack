using OnnxStack.Core.Config;
using OnnxStack.Core.Image;
using OpenCvSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.Core.Video
{
    public static class VideoHelper
    {
        private static OnnxStackConfig _configuration = new OnnxStackConfig();

        /// <summary>
        /// Sets the configuration.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public static void SetConfiguration(OnnxStackConfig configuration)
        {
            _configuration = configuration;
        }


        /// <summary>
        /// Writes the video frames to file.
        /// </summary>
        /// <param name="onnxVideo">The onnx video.</param>
        /// <param name="videoFile">The filename.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public static async Task WriteVideoAsync(string videoFile, OnnxVideo onnxVideo, CancellationToken cancellationToken = default)
        {
            await WriteVideoFramesAsync(videoFile, onnxVideo.Frames, onnxVideo.FrameRate, onnxVideo.Width, onnxVideo.Height, cancellationToken);
        }


        /// <summary>
        /// Writes the video frames to file.
        /// </summary>
        /// <param name="onnxImages">The onnx images.</param>
        /// <param name="videoFile">The filename.</param>
        /// <param name="frameRate">The frame rate.</param>
        /// <param name="aspectRatio">The aspect ratio.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public static async Task WriteVideoFramesAsync(string videoFile, IEnumerable<OnnxImage> onnxImages, float frameRate, int width, int height, CancellationToken cancellationToken = default)
        {
            DeleteFile(videoFile);
            var frames = onnxImages.Select(ImageToMat)
                .ToAsyncEnumerable();
            await WriteFramesInternalAsync(videoFile, frames, frameRate, width, height, cancellationToken);
        }


        /// <summary>
        /// Writes the video stream to file.
        /// </summary>
        /// <param name="onnxImages">The onnx image stream.</param>
        /// <param name="videoFile">The filename.</param>
        /// <param name="frameRate">The frame rate.</param>
        /// <param name="aspectRatio">The aspect ratio.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public static async Task WriteVideoStreamAsync(string videoFile, IAsyncEnumerable<OnnxImage> videoStream, float frameRate, int width, int height, CancellationToken cancellationToken = default)
        {
            DeleteFile(videoFile);
            await WriteFramesInternalAsync(videoFile, videoStream.Select(ImageToMat), frameRate, width, height, cancellationToken);
        }


        /// <summary>
        /// Reads the video information.
        /// </summary>
        /// <param name="videoBytes">The video bytes.</param>
        /// <returns></returns>
        public static async Task<OnnxVideo> ReadVideoInfoAsync(byte[] videoBytes, CancellationToken cancellationToken = default)
        {
            string tempVideoPath = GetTempFilename();
            try
            {
                await File.WriteAllBytesAsync(tempVideoPath, videoBytes, cancellationToken);
                return await ReadVideoInfoAsync(tempVideoPath, cancellationToken);
            }
            finally
            {
                DeleteFile(tempVideoPath);
            }
        }


        /// <summary>
        /// Reads the video information.
        /// </summary>
        /// <param name="videoFile">The filename.</param>
        /// <returns></returns>
        public static async Task<OnnxVideo> ReadVideoInfoAsync(string videoFile, CancellationToken cancellationToken = default)
        {
            return await ReadInfoInternalAsync(videoFile, cancellationToken);
        }


        /// <summary>
        /// Read video frames.
        /// </summary>
        /// <param name="videoBytes">The video bytes.</param>
        /// <param name="frameRate">The desired frame rate.</param>
        /// <param name="width">The desired width. (aspect will be preserved)</param>
        /// <param name="height">The desired height. (aspect will be preserved)</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;System.Collections.Generic.List<OnnxStack.Core.Image.OnnxImage>&gt; representing the asynchronous operation.</returns>
        public static async Task<OnnxVideo> ReadVideoAsync(byte[] videoBytes, float? frameRate = default, int? width = default, int? height = default, CancellationToken cancellationToken = default)
        {
            string tempVideoPath = GetTempFilename();
            try
            {
                await File.WriteAllBytesAsync(tempVideoPath, videoBytes, cancellationToken);
                return await ReadVideoAsync(tempVideoPath, frameRate, width, height, cancellationToken);
            }
            finally
            {
                DeleteFile(tempVideoPath);
            }
        }



        /// <summary>
        /// Reads the video frames.
        /// </summary>
        /// <param name="videoFile">The video bytes.</param>
        /// <param name="frameRate">The desired frame rate.</param>
        /// <param name="width">The desired width. (aspect will be preserved)</param>
        /// <param name="height">The desired height. (aspect will be preserved)</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public static async Task<OnnxVideo> ReadVideoAsync(string videoFile, float? frameRate = default, int? width = default, int? height = default, CancellationToken cancellationToken = default)
        {
            var frames = await ReadFramesInternalAsync(videoFile, frameRate, width, height, cancellationToken).ToArrayAsync(cancellationToken);
            return new OnnxVideo(ToImageFrames(frames, cancellationToken), frames[0].FrameRate);
        }


        /// <summary>
        /// Read video frames
        /// </summary>
        /// <param name="videoFile">The video file.</param>
        /// <param name="frameRate">The desired frame rate.</param>
        /// <param name="width">The desired width. (aspect will be preserved)</param>
        /// <param name="height">The desired height. (aspect will be preserved)</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;List`1&gt; representing the asynchronous operation.</returns>
        public static async Task<List<OnnxImage>> ReadVideoFramesAsync(string videoFile, float? frameRate = default, int? width = default, int? height = default, CancellationToken cancellationToken = default)
        {
            var frames = await ReadFramesInternalAsync(videoFile, frameRate, width, height, cancellationToken).ToArrayAsync(cancellationToken);
            return ToImageFrames(frames, cancellationToken);
        }


        /// <summary>
        /// Reads the video frames as a stream.
        /// </summary>
        /// <param name="videoFile">The filename.</param>
        /// <param name="frameRate">The desired frame rate.</param>
        /// <param name="width">The desired width. (aspect will be preserved)</param>
        /// <param name="height">The desired height. (aspect will be preserved)</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public static async IAsyncEnumerable<OnnxImage> ReadVideoStreamAsync(string videoFile, float? frameRate = default, int? width = default, int? height = default, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await foreach (var frame in ReadFramesInternalAsync(videoFile, frameRate, width, height, cancellationToken))
            {
                using (var image = MatToImage(frame.Mat))
                {
                    yield return new OnnxImage(image);
                }
            }
        }


        /// <summary>
        /// Reads the device video frames as a stream.
        /// </summary>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;IAsyncEnumerable`1&gt; representing the asynchronous operation.</returns>
        public static async IAsyncEnumerable<OnnxImage> ReadVideoStreamAsync(int deviceId, int? width = default, int? height = default, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await foreach (var frame in ReadFramesInternalAsync(deviceId, width, height, cancellationToken))
            {
                using (var image = MatToImage(frame.Mat))
                {
                    yield return new OnnxImage(image);
                }
            }
        }


        /// <summary>
        /// Reads the video information
        /// </summary>
        /// <param name="videoFile">The video path.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>Task&lt;VideoInfo&gt;.</returns>
        /// <exception cref="System.Exception">Failed to open video file.</exception>
        private static async Task<OnnxVideo> ReadInfoInternalAsync(string videoFile, CancellationToken cancellationToken = default)
        {
            return await Task.Run(() =>
            {
                using (var videoReader = new VideoCapture(videoFile))
                {
                    if (!videoReader.IsOpened())
                        throw new Exception("Failed to open video file.");

                    var width = videoReader.FrameWidth;
                    var height = videoReader.FrameHeight;
                    var frameRate = (float)videoReader.Fps;
                    var frameCount = videoReader.FrameCount;
                    return Task.FromResult(new OnnxVideo(frameCount, frameRate, width, height));
                }
            }, cancellationToken);
        }


        /// <summary>
        /// Read the video frames
        /// </summary>
        /// <param name="videoFile">The video path.</param>
        /// <param name="frameRate">The desired frame rate.</param>
        /// <param name="width">The desired width. (aspect will be preserved)</param>
        /// <param name="height">The desired height. (aspect will be preserved)</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;IAsyncEnumerable`1&gt; representing the asynchronous operation.</returns>
        /// <exception cref="System.Exception">Failed to open video file.</exception>
        private static async IAsyncEnumerable<MatFrame> ReadFramesInternalAsync(string videoFile, float? frameRate = default, int? width = default, int? height = default, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            using (var videoReader = new VideoCapture(videoFile))
            {
                if (!videoReader.IsOpened())
                    throw new Exception("Failed to open video file.");

                var frameCount = 0;
                var emptySize = new OpenCvSharp.Size(0, 0);
                var outframeRate = (float)(frameRate.HasValue ? Math.Min(frameRate.Value, videoReader.Fps) : videoReader.Fps);
                var frameSkipInterval = frameRate.HasValue ? (int)Math.Round(videoReader.Fps / Math.Min(frameRate.Value, videoReader.Fps)) : 1;
                var isScaleRequired = width.HasValue || height.HasValue;
                var scaleX = (width ?? videoReader.FrameWidth) / (double)videoReader.FrameWidth;
                var scaleY = (height ?? videoReader.FrameHeight) / (double)videoReader.FrameHeight;
                var scaleFactor = scaleX < 1 && scaleY < 1 ? Math.Max(scaleX, scaleY) : Math.Min(scaleX, scaleY);
                using (var frame = new Mat())
                {
                    while (true)
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        videoReader.Read(frame);
                        if (frame.Empty())
                            break;

                        if (frameCount % frameSkipInterval == 0)
                        {
                            if (isScaleRequired)
                                Cv2.Resize(frame, frame, emptySize, scaleFactor, scaleFactor);

                            yield return new MatFrame(frame.Clone(), outframeRate);
                        }

                        frameCount++;
                    }
                }
                await Task.Yield();
            }
        }


        private static async IAsyncEnumerable<MatFrame> ReadFramesInternalAsync(int deviceId, int? width = default, int? height = default, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            using (var videoReader = new VideoCapture(deviceId, VideoCaptureAPIs.DSHOW))
            {
                if (!videoReader.IsOpened())
                    throw new Exception($"Failed to open device {deviceId}.");

                var emptySize = new OpenCvSharp.Size(0, 0);
                var isScaleRequired = width.HasValue || height.HasValue;
                var scaleX = (width ?? videoReader.FrameWidth) / (double)videoReader.FrameWidth;
                var scaleY = (height ?? videoReader.FrameHeight) / (double)videoReader.FrameHeight;
                var scaleFactor = scaleX < 1 && scaleY < 1 ? Math.Max(scaleX, scaleY) : Math.Min(scaleX, scaleY);
                using (var frame = new Mat())
                {
                    while (true)
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        videoReader.Read(frame);
                        if (frame.Empty())
                            break;

                        if (isScaleRequired)
                            Cv2.Resize(frame, frame, emptySize, scaleFactor, scaleFactor);

                        yield return new MatFrame(frame.Clone(), 0);
                    }
                }
                await Task.Yield();
            }
        }


        /// <summary>
        /// Write the video frames
        /// </summary>
        /// <param name="videoFile">The output path.</param>
        /// <param name="frames">The frames.</param>
        /// <param name="framerate">The framerate.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task representing the asynchronous operation.</returns>
        /// <exception cref="System.Exception">Failed to open VideoWriter..</exception>
        private static async Task WriteFramesInternalAsync(string videoFile, IAsyncEnumerable<Mat> frames, float framerate, int width, int height, CancellationToken cancellationToken = default)
        {
            var fourcc = VideoWriter.FourCC(_configuration.VideoCodec);
            var frameSize = new OpenCvSharp.Size(width, height);
            await Task.Run(async () =>
            {
                using (var writer = new VideoWriter(videoFile, fourcc, framerate, frameSize))
                {
                    if (!writer.IsOpened())
                        throw new Exception("Failed to open VideoWriter..");

                    await foreach (var frame in frames)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        writer.Write(frame);
                        frame.Dispose();
                    }
                }
            }, cancellationToken);
        }


        /// <summary>
        /// Gets the temporary filename.
        /// </summary>
        /// <returns></returns>
        private static string GetTempFilename()
        {
            if (!Directory.Exists(_configuration.TempPath))
                Directory.CreateDirectory(_configuration.TempPath);

            return Path.Combine(_configuration.TempPath, $"{Path.GetFileNameWithoutExtension(Path.GetRandomFileName())}.mp4");
        }


        /// <summary>
        /// Deletes the temporary file.
        /// </summary>
        /// <param name="filename">The filename.</param>
        private static void DeleteFile(string filename)
        {
            try
            {
                if (File.Exists(filename))
                    File.Delete(filename);
            }
            catch (Exception)
            {
                // File in use, Log
            }
        }


        /// <summary>
        /// OnnxImage to Mat.
        /// </summary>
        /// <param name="image">The image.</param>
        /// <returns>Mat.</returns>
        public static Mat ImageToMat(OnnxImage image)
        {
            return ImageToMat(image.GetImage());
        }


        /// <summary>
        /// ImageSharp to Mat.
        /// </summary>
        /// <param name="image">The image.</param>
        /// <returns>Mat.</returns>
        private static Mat ImageToMat(Image<Rgba32> image)
        {
            var mat = new Mat(image.Height, image.Width, MatType.CV_8UC3);
            for (int y = 0; y < image.Height; y++)
            {
                var pixelRow = image.DangerousGetPixelRowMemory(y).Span;
                for (int x = 0; x < image.Width; x++)
                {
                    var pixel = pixelRow[x];
                    mat.Set(y, x, new Vec3b(pixel.B, pixel.G, pixel.R));
                }
            }
            return mat;
        }


        /// <summary>
        /// Mat to ImageSharp
        /// </summary>
        /// <param name="mat">The mat.</param>
        /// <returns>Image&lt;Rgba32&gt;.</returns>
        private static Image<Rgba32> MatToImage(Mat mat)
        {
            var image = new Image<Rgba32>(mat.Width, mat.Height);
            for (int y = 0; y < mat.Rows; y++)
            {
                var pixelRow = image.DangerousGetPixelRowMemory(y).Span;
                for (int x = 0; x < mat.Cols; x++)
                {
                    var vec = mat.Get<Vec3b>(y, x);
                    pixelRow[x] = new Rgba32(vec.Item2, vec.Item1, vec.Item0, 255);
                }
            }
            return image;
        }


        /// <summary>
        /// Converts to imageframes.
        /// </summary>
        /// <param name="frames">The frames.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>List&lt;OnnxImage&gt;.</returns>
        private static List<OnnxImage> ToImageFrames(MatFrame[] frames, CancellationToken cancellationToken)
        {
            var results = new List<OnnxImage>(new OnnxImage[frames.Length]);
            Parallel.For(0, frames.Length, (i) =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                using (var frame = frames[i].Mat)
                using (var image = MatToImage(frame))
                {
                    results[i] = new OnnxImage(image);
                }
            });
            return results;
        }


        /// <summary>
        /// MatFrame
        /// </summary>
        private record struct MatFrame(Mat Mat, float FrameRate);
    }
}
