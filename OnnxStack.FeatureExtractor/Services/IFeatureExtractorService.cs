using OnnxStack.Core.Image;
using System.Threading.Tasks;

namespace OnnxStack.FeatureExtractor.Common
{
    public interface IFeatureExtractorService
    {
        Task<InputImage> CannyImage(FeatureExtractorModelSet controlNetModel, InputImage inputImage, int height, int width);
        Task<InputImage> HedImage(FeatureExtractorModelSet controlNetModel, InputImage inputImage, int height, int width);
        Task<InputImage> DepthImage(FeatureExtractorModelSet controlNetModel, InputImage inputImage, int height, int width);
    }
}