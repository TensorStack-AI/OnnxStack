using OnnxStack.Core.Image;
using OnnxStack.Core.Video;
using OnnxStack.FeatureExtractor.Common;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.UI.Services
{
    public interface IFeatureExtractorService
    {

        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        Task<bool> LoadModelAsync(FeatureExtractorModelSet model);

        /// <summary>
        /// Unloads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        Task<bool> UnloadModelAsync(FeatureExtractorModelSet model);

        /// <summary>
        /// Determines whether [is model loaded] [the specified model options].
        /// </summary>
        /// <param name="model">The modelset.</param>
        /// <returns>
        ///   <c>true</c> if [is model loaded] [the specified model options]; otherwise, <c>false</c>.
        /// </returns>
        bool IsModelLoaded(FeatureExtractorModelSet model);

        /// <summary>
        /// Generates the feature image.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        Task<OnnxImage> GenerateAsync(FeatureExtractorModelSet model, OnnxImage inputImage, CancellationToken cancellationToken = default);

        /// <summary>
        /// Generates the feature video.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="inputVideo">The input video.</param>
        /// <returns></returns>
        Task<OnnxVideo> GenerateAsync(FeatureExtractorModelSet model, OnnxVideo inputVideo, CancellationToken cancellationToken = default);
    }
}
