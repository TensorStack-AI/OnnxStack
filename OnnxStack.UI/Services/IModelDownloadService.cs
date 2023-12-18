using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.UI.Services
{
    public interface IModelDownloadService
    {

        /// <summary>
        /// Downloads the model via HTTP.
        /// </summary>
        /// <param name="modelFileList">The model files to download.</param>
        /// <param name="destinationPath">The destination path.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
       // Task<bool> DownloadHttpAsync(ModelConfigTemplate modelConfigTemplate, string destinationPath, Action<string, double, double> progressCallback = null, CancellationToken cancellationToken = default);
        Task<bool> DownloadHttpAsync(List<string> modelFileList, string destinationPath, Action<string, double, double> progressCallback = null, CancellationToken cancellationToken = default, double increment = 1);


        /// <summary>
        /// Downloads the model repository.
        /// </summary>
        /// <param name="repositoryUrl">The repository Url.</param>
        /// <param name="destinationPath">The destination path.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        Task<bool> DownloadRepositoryAsync(string repositoryUrl, string destinationPath, Action<string, double, double> progressCallback = null, CancellationToken cancellationToken = default);

    }
}