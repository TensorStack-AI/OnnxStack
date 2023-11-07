using OnnxStack.UI.Views;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.UI.Services
{
    public interface IModelDownloadService
    {

        /// <summary>
        /// Downloads the model via HTTP.
        /// </summary>
        /// <param name="modelConfigTemplate">The model configuration template.</param>
        /// <param name="destinationPath">The destination path.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        Task<bool> DownloadHttp(ModelConfigTemplate modelConfigTemplate, string destinationPath, Action<string, double, double> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Downloads the model repository.
        /// </summary>
        /// <param name="modelConfigTemplate">The model configuration template.</param>
        /// <param name="destinationPath">The destination path.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        Task<bool> DownloadRepository(ModelConfigTemplate modelConfigTemplate, string destinationPath, Action<string, double, double> progressCallback = null, CancellationToken cancellationToken = default);
    }
}