using OnnxStack.Core.Image;
using OnnxStack.StableDiffusion.Config;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IImageService
    {

        /// <summary>
        /// Prepares the ContolNet input image, If the ControlNetModelSet has a configure Annotation model this will be used to process the image
        /// </summary>
        /// <param name="controlNetModel">The control net model.</param>
        /// <param name="inputImage">The input image.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <returns></returns>
        Task<InputImage> PrepareInputImage(ControlNetModelSet controlNetModel, InputImage inputImage, int height, int width);
    }
}