using FFMpegCore;
using OnnxStack.Core.Config;
using OnnxStack.Core.Image;
using System;
using System.Collections.Generic;
using System.Diagnostics;
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
        /// <param name="filename">The filename.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public static async Task WriteVideoFramesAsync(OnnxVideo onnxVideo, string filename, CancellationToken cancellationToken = default)
        {
            await WriteVideoFramesAsync(onnxVideo.Frames, filename, onnxVideo.FrameRate, onnxVideo.AspectRatio, cancellationToken);
        }


        /// <summary>
        /// Writes the video frames to file.
        /// </summary>
        /// <param name="onnxImages">The onnx images.</param>
        /// <param name="filename">The filename.</param>
        /// <param name="frameRate">The frame rate.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public static async Task WriteVideoFramesAsync(IEnumerable<OnnxImage> onnxImages, string filename, float frameRate = 15, CancellationToken cancellationToken = default)
        {
            var firstImage = onnxImages.First();
            var aspectRatio = (double)firstImage.Width / firstImage.Height;
            await WriteVideoFramesAsync(onnxImages, filename, frameRate, aspectRatio, cancellationToken);
        }


        /// <summary>
        /// Writes the video frames to file.
        /// </summary>
        /// <param name="onnxImages">The onnx images.</param>
        /// <param name="filename">The filename.</param>
        /// <param name="frameRate">The frame rate.</param>
        /// <param name="aspectRatio">The aspect ratio.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private static async Task WriteVideoFramesAsync(IEnumerable<OnnxImage> onnxImages, string filename, float frameRate, double aspectRatio, CancellationToken cancellationToken = default)
        {
            if (File.Exists(filename))
                File.Delete(filename);

            using (var videoWriter = CreateWriter(filename, frameRate, aspectRatio))
            {
                // Start FFMPEG
                videoWriter.Start();
                foreach (var image in onnxImages)
                {
                    // Write each frame to the input stream of FFMPEG
                    await videoWriter.StandardInput.BaseStream.WriteAsync(image.GetImageBytes(), cancellationToken);
                }

                // Done close stream and wait for app to process
                videoWriter.StandardInput.BaseStream.Close();
                await videoWriter.WaitForExitAsync(cancellationToken);
            }
        }


        /// <summary>
        /// Writes the video stream to file.
        /// </summary>
        /// <param name="onnxImages">The onnx image stream.</param>
        /// <param name="filename">The filename.</param>
        /// <param name="frameRate">The frame rate.</param>
        /// <param name="aspectRatio">The aspect ratio.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public static async Task WriteVideoStreamAsync(VideoInfo videoInfo, IAsyncEnumerable<OnnxImage> videoStream, string filename, CancellationToken cancellationToken = default)
        {
            if (File.Exists(filename))
                File.Delete(filename);

            using (var videoWriter = CreateWriter(filename, videoInfo.FrameRate, videoInfo.AspectRatio))
            {
                // Start FFMPEG
                videoWriter.Start();
                await foreach (var frame in videoStream)
                {
                    // Write each frame to the input stream of FFMPEG
                    await videoWriter.StandardInput.BaseStream.WriteAsync(frame.GetImageBytes(), cancellationToken);
                }

                // Done close stream and wait for app to process
                videoWriter.StandardInput.BaseStream.Close();
                await videoWriter.WaitForExitAsync(cancellationToken);
            }
        }


        /// <summary>
        /// Reads the video information.
        /// </summary>
        /// <param name="videoBytes">The video bytes.</param>
        /// <returns></returns>
        public static async Task<VideoInfo> ReadVideoInfoAsync(byte[] videoBytes)
        {
            using (var memoryStream = new MemoryStream(videoBytes))
            {
                var result = await FFProbe.AnalyseAsync(memoryStream).ConfigureAwait(false);
                return new VideoInfo(result.PrimaryVideoStream.Width, result.PrimaryVideoStream.Height, result.Duration, (int)result.PrimaryVideoStream.FrameRate);
            }
        }


        /// <summary>
        /// Reads the video information.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <returns></returns>
        public static async Task<VideoInfo> ReadVideoInfoAsync(string filename)
        {
            var result = await FFProbe.AnalyseAsync(filename).ConfigureAwait(false);
            return new VideoInfo(result.PrimaryVideoStream.Width, result.PrimaryVideoStream.Height, result.Duration, (int)result.PrimaryVideoStream.FrameRate);
        }


        /// <summary>
        /// Reads the video frames.
        /// </summary>
        /// <param name="videoBytes">The video bytes.</param>
        /// <param name="frameRate">The target frame rate.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public static async Task<List<OnnxImage>> ReadVideoFramesAsync(byte[] videoBytes, float frameRate = 15, CancellationToken cancellationToken = default)
        {
            string tempVideoPath = GetTempFilename();
            try
            {
                await File.WriteAllBytesAsync(tempVideoPath, videoBytes, cancellationToken);
                return await ReadVideoStreamAsync(tempVideoPath, frameRate, cancellationToken).ToListAsync(cancellationToken);
            }
            finally
            {
                DeleteTempFile(tempVideoPath);
            }
        }


        /// <summary>
        /// Reads the video frames.
        /// </summary>
        /// <param name="filename">The video bytes.</param>
        /// <param name="frameRate">The target frame rate.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public static async Task<List<OnnxImage>> ReadVideoFramesAsync(string filename, float frameRate = 15, CancellationToken cancellationToken = default)
        {
            return await ReadVideoStreamAsync(filename, frameRate, cancellationToken).ToListAsync(cancellationToken);
        }


        /// <summary>
        /// Reads the video frames as a stream.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="frameRate">The frame rate.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public static async IAsyncEnumerable<OnnxImage> ReadVideoStreamAsync(string filename, float frameRate = 15, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await foreach (var frameBytes in CreateFramesInternalAsync(filename, frameRate, cancellationToken))
            {
                yield return new OnnxImage(frameBytes);
            }
        }


        #region Private Members


        /// <summary>
        /// Creates a collection of PNG frames from a video source
        /// </summary>
        /// <param name="videoData">The video data.</param>
        /// <param name="fps">The FPS.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        /// <exception cref="Exception">Invalid PNG header</exception>
        private static async IAsyncEnumerable<byte[]> CreateFramesInternalAsync(string fileName, float fps = 15, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            using (var ffmpegProcess = CreateReader(fileName, fps))
            {
                // Start FFMPEG
                ffmpegProcess.Start();

                // FFMPEG output stream
                var processOutputStream = ffmpegProcess.StandardOutput.BaseStream;

                // Buffer to hold the current image
                var buffer = new byte[20480000];

                var currentIndex = 0;
                while (!cancellationToken.IsCancellationRequested)
                {
                    // Reset the index new PNG
                    currentIndex = 0;

                    // Read the PNG Header
                    if (await processOutputStream.ReadAsync(buffer.AsMemory(currentIndex, 8), cancellationToken) <= 0)
                        break;

                    currentIndex += 8;// header length

                    if (!IsImageHeader(buffer))
                        throw new Exception("Invalid PNG header");

                    // loop through each chunk
                    while (true)
                    {
                        // Read the chunk header
                        await processOutputStream.ReadAsync(buffer.AsMemory(currentIndex, 12), cancellationToken);

                        var chunkIndex = currentIndex;
                        currentIndex += 12; // Chunk header length

                        // Get the chunk's content size in bytes from the header we just read
                        var totalSize = buffer[chunkIndex] << 24 | buffer[chunkIndex + 1] << 16 | buffer[chunkIndex + 2] << 8 | buffer[chunkIndex + 3];
                        if (totalSize > 0)
                        {
                            var totalRead = 0;
                            while (totalRead < totalSize)
                            {
                                int read = await processOutputStream.ReadAsync(buffer.AsMemory(currentIndex, totalSize - totalRead), cancellationToken);
                                currentIndex += read;
                                totalRead += read;
                            }
                            continue;
                        }

                        // If the size is 0 and is the end of the image
                        if (totalSize == 0 && IsImageEnd(buffer, chunkIndex))
                            break;
                    }

                    yield return buffer[..currentIndex];
                }

                if (cancellationToken.IsCancellationRequested)
                    ffmpegProcess.Kill();
            }
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
        private static void DeleteTempFile(string filename)
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
        /// Creates FFMPEG video reader process.
        /// </summary>
        /// <param name="inputFile">The input file.</param>
        /// <param name="fps">The FPS.</param>
        /// <returns></returns>
        private static Process CreateReader(string inputFile, float fps)
        {
            var ffmpegProcess = new Process();
            ffmpegProcess.StartInfo.FileName = _configuration.FFmpegPath;
            ffmpegProcess.StartInfo.Arguments = $"-hide_banner -loglevel error -i \"{inputFile}\" -c:v png  -r {fps} -f image2pipe -";
            ffmpegProcess.StartInfo.RedirectStandardOutput = true;
            ffmpegProcess.StartInfo.UseShellExecute = false;
            ffmpegProcess.StartInfo.CreateNoWindow = true;
            return ffmpegProcess;
        }


        /// <summary>
        /// Creates FFMPEG video writer process.
        /// </summary>
        /// <param name="outputFile">The output file.</param>
        /// <param name="fps">The FPS.</param>
        /// <param name="aspectRatio">The aspect ratio.</param>
        /// <returns></returns>
        private static Process CreateWriter(string outputFile, float fps, double aspectRatio)
        {
            var ffmpegProcess = new Process();
            ffmpegProcess.StartInfo.FileName = _configuration.FFmpegPath;
            ffmpegProcess.StartInfo.Arguments = $"-hide_banner -loglevel error -framerate {fps:F4} -i - -c:v libx264 -movflags +faststart -vf format=yuv420p -aspect {aspectRatio} {outputFile}";
            ffmpegProcess.StartInfo.RedirectStandardInput = true;
            ffmpegProcess.StartInfo.UseShellExecute = false;
            ffmpegProcess.StartInfo.CreateNoWindow = true;
            return ffmpegProcess;
        }


        /// <summary>
        /// Determines whether we are at the start of a PNG image in the specified buffer.
        /// </summary>
        /// <param name="buffer">The buffer.</param>
        /// <param name="offset">The offset.</param>
        /// <returns>
        ///   <c>true</c> if the start of a PNG image sequence is detected<c>false</c>.
        /// </returns>
        private static bool IsImageHeader(byte[] buffer)
        {
            // PNG Header http://www.libpng.org/pub/png/spec/1.2/PNG-Structure.html#PNG-file-signature
            if (buffer[0] != 0x89
             || buffer[1] != 0x50
             || buffer[2] != 0x4E
             || buffer[3] != 0x47
             || buffer[4] != 0x0D
             || buffer[5] != 0x0A
             || buffer[6] != 0x1A
             || buffer[7] != 0x0A)
                return false;

            return true;
        }


        /// <summary>
        /// Determines whether we are at the end of a PNG image in the specified buffer.
        /// </summary>
        /// <param name="buffer">The buffer.</param>
        /// <param name="offset">The offset.</param>
        /// <returns>
        ///   <c>true</c> if the end of a PNG image sequence is detected<c>false</c>.
        /// </returns>
        private static bool IsImageEnd(byte[] buffer, int offset)
        {
            return buffer[offset + 4] == 0x49  // I
                && buffer[offset + 5] == 0x45  // E
                && buffer[offset + 6] == 0x4E  // N
                && buffer[offset + 7] == 0x44; // D
        }
    }

    #endregion
}
