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
        /// Downloads the model via HTTP.
        /// </summary>
        /// <param name="modelConfigTemplate">The model configuration template.</param>
        /// <param name="destinationPath">The destination path.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async Task<bool> DownloadHttp(ModelConfigTemplate modelConfigTemplate, string destinationPath, Action<string, double, double> progressCallback = default, CancellationToken cancellationToken = default)
        {
            return await DownloadFileAsync(modelConfigTemplate.ModelFiles, destinationPath, progressCallback, cancellationToken);
        }


        /// <summary>
        /// Downloads the model repository.
        /// </summary>
        /// <param name="modelConfigTemplate">The model configuration template.</param>
        /// <param name="destinationPath">The destination path.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async Task<bool> DownloadRepository(ModelConfigTemplate modelConfigTemplate, string destinationPath, Action<string, double, double> progressCallback = default, CancellationToken cancellationToken = default)
        {
            return await DownloadRepositoryAsync(modelConfigTemplate.Repository, destinationPath, progressCallback, cancellationToken);
        }


        /// <summary>
        /// Downloads the repository.
        /// </summary>
        /// <param name="repository">The repository.</param>
        /// <param name="destinationPath">The destination path.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        private async Task<bool> DownloadRepositoryAsync(string repository, string destinationPath, Action<string, double, double> progressCallback = default, CancellationToken cancellationToken = default)
        {

            using (var downloader = new RepositoryDownloader(repository, destinationPath, (f, p) => progressCallback?.Invoke(f, p, p)))
            {
                await downloader.DownloadAsync();
            }
            return true;
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
        private async Task<bool> DownloadFileAsync(List<string> fileList, string output, Action<string, double, double> progressCallback, CancellationToken cancellationToken = default, double increment = 1)
        {
            using (var httpClient = new HttpClient())
            {
                var totalDownloadSize = await GetTotalSizeFromHeaders(fileList, httpClient);
                if (totalDownloadSize == 0)
                    throw new Exception("Queried file headers returned 0 bytes");

                var totalBytesRead = 0L;
                foreach (var file in fileList)
                {
                    var filename = Path.GetFileName(file);
                    var directory = Path.GetDirectoryName(file).Split(new[] { '\\', '/' }).LastOrDefault();
                    var destination = Path.Combine(output, directory);
                    var destinationFile = Path.Combine(destination, filename);

                    try
                    {
                        Directory.CreateDirectory(destination);
                        using (var response = await httpClient.GetAsync(file, HttpCompletionOption.ResponseHeadersRead))
                        {
                            response.EnsureSuccessStatusCode();
                            var fileSize = response.Content.Headers.ContentLength ?? -1;
                            var canReportProgress = fileSize != -1 && progressCallback != null;
                            var buffer = new byte[8192];
                            var bytesRead = 0;

                            var lastProgress = 0d;
                            using (var fileStream = File.Create(destinationFile))
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
                                            progressCallback?.Invoke(file, fileProgress, totalProgressValue);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        TryDelete(destinationFile);
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
        private static async Task<long> GetTotalSizeFromHeaders(List<string> fileList, HttpClient httpClient)
        {
            try
            {
                var totalDownloadSize = 0L;
                foreach (var file in fileList)
                {
                    using (var response = await httpClient.GetAsync(file, HttpCompletionOption.ResponseHeadersRead))
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
    }
}
