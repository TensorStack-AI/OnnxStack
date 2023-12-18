using OnnxStack.UI.Helpers;
using OnnxStack.UI.Views;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.UI.Services
{
    public class ModelDownloadService : IModelDownloadService
    {
        /// <summary>
        /// Downloads the repository.
        /// </summary>
        /// <param name="repository">The repository.</param>
        /// <param name="destinationPath">The destination path.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async Task<bool> DownloadRepositoryAsync(string repositoryUrl, string destinationPath, Action<string, double, double> progressCallback = default, CancellationToken cancellationToken = default)
        {
            using (var downloader = new RepositoryDownloader(repositoryUrl, destinationPath, (f, p) => progressCallback?.Invoke(f, p, p)))
            {
                return await downloader.DownloadAsync(cancellationToken);
            }
        }


        /// <summary>
        /// Downloads the files.
        /// </summary>
        /// <param name="fileList">The file list.</param>
        /// <param name="output">The output.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <param name="increment">The increment.</param>
        /// <returns></returns>
        public async Task<bool> DownloadHttpAsync(List<string> modelFileList, string output, Action<string, double, double> progressCallback, CancellationToken cancellationToken = default, double increment = 1)
        {
            var remainingFiles = GetFiles(modelFileList, output);
            using (var httpClient = new HttpClient())
            {
                var totalDownloadSize = await GetTotalSizeFromHeadersAsync(remainingFiles, httpClient);
                if (totalDownloadSize == 0)
                    throw new Exception("Queried file headers returned 0 bytes");

                var totalBytesRead = 0L;
                foreach (var file in remainingFiles)
                {
                    try
                    {
                        using (var response = await httpClient.GetAsync(file.Url, HttpCompletionOption.ResponseHeadersRead))
                        {
                            response.EnsureSuccessStatusCode();
                            var fileSize = response.Content.Headers.ContentLength ?? -1;
                            var canReportProgress = fileSize != -1 && progressCallback != null;
                            var buffer = new byte[8192];
                            var bytesRead = 0;

                            var lastProgress = 0d;
                            using (var fileStream = File.Create(file.FileName))
                            using (var stream = await response.Content.ReadAsStreamAsync())
                            {
                                while (true)
                                {
                                    cancellationToken.ThrowIfCancellationRequested();

                                    var readSize = await stream.ReadAsync(buffer, 0, buffer.Length);
                                    if (readSize == 0)
                                        break;

                                    await fileStream.WriteAsync(buffer, 0, readSize);
                                    totalBytesRead += readSize;
                                    bytesRead += readSize;

                                    if (canReportProgress)
                                    {
                                        var fileProgress = Math.Round((bytesRead * 100.0 / fileSize), 3);
                                        var totalProgressValue = Math.Round((totalBytesRead * 100.0 / totalDownloadSize), 3);
                                        if (totalProgressValue > lastProgress || totalProgressValue >= 100)
                                        {
                                            lastProgress = totalProgressValue + increment;
                                            progressCallback?.Invoke(file.Url, fileProgress, totalProgressValue);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        TryDelete(file.FileName);
                        throw new Exception($"Error: {ex.Message}");
                    }
                }
                return true;
            }
        }


        /// <summary>
        /// Gets the total size from headers.
        /// </summary>
        /// <param name="fileList">The file list.</param>
        /// <param name="httpClient">The HTTP client.</param>
        /// <returns></returns>
        /// <exception cref="Exception">Failed to query file headers, {ex.Message}</exception>
        private static async Task<long> GetTotalSizeFromHeadersAsync(IEnumerable<FileInfo> fileList, HttpClient httpClient)
        {
            try
            {
                var totalDownloadSize = 0L;
                foreach (var file in fileList)
                {
                    using (var response = await httpClient.GetAsync(file.Url, HttpCompletionOption.ResponseHeadersRead))
                    {
                        response.EnsureSuccessStatusCode();
                        totalDownloadSize += response.Content.Headers.ContentLength ?? 0;
                    }
                }
                return totalDownloadSize;
            }
            catch (Exception ex)
            {
                throw new Exception($"Failed to query file headers, {ex.Message}");
            }
        }


        /// <summary>
        /// Tries the delete.
        /// </summary>
        /// <param name="filename">The filename.</param>
        private void TryDelete(string filename)
        {
            try
            {
                if (File.Exists(filename))
                    File.Delete(filename);
            }
            catch (Exception)
            {
                // LOG ME
            }
        }

        private IEnumerable<FileInfo> GetFiles(IEnumerable<string> urlFileList, string outputDirectory)
        {
            foreach (var fileUrl in urlFileList)
            {
                var filename = Path.GetFileName(fileUrl);
                var directory = Path.GetDirectoryName(fileUrl).Split(new[] { '\\', '/' }).LastOrDefault();
                var destination = Path.Combine(outputDirectory, directory);
                var destinationFile = Path.Combine(destination, filename);
                if (File.Exists(destinationFile))
                    continue;

                Directory.CreateDirectory(destination);
                yield return new FileInfo(fileUrl, destinationFile);
            }
        }

        private record FileInfo(string Url, string FileName);
    }
}
