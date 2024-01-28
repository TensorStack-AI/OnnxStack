using OnnxStack.Core.Image;
using System.Threading.Tasks;

namespace OnnxStack.FeatureExtractor.Common
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
        /// <param name="modelOptions">The model options.</param>
        /// <returns>
        ///   <c>true</c> if [is model loaded] [the specified model options]; otherwise, <c>false</c>.
        /// </returns>
        bool IsModelLoaded(FeatureExtractorModelSet modelOptions);

        Task<InputImage> CannyImage(FeatureExtractorModelSet controlNetModel, InputImage inputImage, int height, int width);
        Task<InputImage> HedImage(FeatureExtractorModelSet controlNetModel, InputImage inputImage, int height, int width);
        Task<InputImage> DepthImage(FeatureExtractorModelSet controlNetModel, InputImage inputImage, int height, int width);
    }
}