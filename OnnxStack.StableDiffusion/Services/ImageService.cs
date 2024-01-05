using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using System.Linq;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Services
{
    /// <summary>
    /// Service for handing images for input and output of the diffusion process
    /// </summary>
    /// <seealso cref="OnnxStack.StableDiffusion.Common.IImageService" />
    public class ImageService : IImageService
    {
        private readonly ILogger<ImageService> _logger;
        private readonly IOnnxModelService _onnxModelService;


        /// <summary>
        /// Initializes a new instance of the <see cref="ImageService"/> class.
        /// </summary>
        /// <param name="onnxModelService">The onnx model service.</param>
        public ImageService(IOnnxModelService onnxModelService, ILogger<ImageService> logger)
        {
            _logger = logger;
            _onnxModelService = onnxModelService;
        }


        /// <summary>
        /// Prepares the ControlNet input image.
        /// </summary>
        /// <param name="controlNetModel">The control net model.</param>
        /// <param name="inputImage">The input image.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <returns></returns>
        public async Task<InputImage> PrepareInputImage(ControlNetModelSet controlNetModel, InputImage inputImage, int height, int width)
        {
            var annotationModel = controlNetModel.ModelConfigurations.FirstOrDefault(x => x.Type == OnnxModelType.Annotation);
            if (annotationModel is not null)
            {
                _logger.LogInformation($"[PrepareInputImage] - ControlNet {controlNetModel.Type} annotation model found.");
                return controlNetModel.Type switch
                {
                    ControlNetType.Canny => await GenerateCannyImage(controlNetModel, inputImage, height, width),
                    ControlNetType.HED => await GenerateHardEdgeImage(controlNetModel, inputImage, height, width),
                    ControlNetType.Depth => await GenerateDepthImage(controlNetModel, inputImage, height, width),
                    _ => PrepareInputImage(inputImage, height, width)
                };
            }

            return PrepareInputImage(inputImage, height, width);
        }


        /// <summary>
        /// Generates the canny image mask.
        /// </summary>
        /// <param name="controlNetModel">The control net model.</param>
        /// <param name="inputImage">The input image.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <returns></returns>
        private async Task<InputImage> GenerateCannyImage(ControlNetModelSet controlNetModel, InputImage inputImage, int height, int width)
        {
            _logger.LogInformation($"[GenerateCannyImage] - Generating Canny image...");
            var controlImage = inputImage.ToDenseTensor(new[] { 1, 3, height, width }, false);
            var metadata = _onnxModelService.GetModelMetadata(controlNetModel, OnnxModelType.Annotation);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(controlImage);
                inferenceParameters.AddOutputBuffer(new[] { 1, 1, height, width });

                var results = await _onnxModelService.RunInferenceAsync(controlNetModel, OnnxModelType.Annotation, inferenceParameters);
                using (var result = results.First())
                {
                    var testImage = result.ToDenseTensor().Repeat(3);
                    var imageTensor = new DenseTensor<float>(controlImage.Dimensions);
                    for (int i = 0; i < testImage.Length; i++)
                        imageTensor.SetValue(i, testImage.GetValue(i));

                    var maskImage = imageTensor.ToImageMask();
                    //await maskImage.SaveAsPngAsync("D:\\Canny.png");
                    _logger.LogInformation($"[GenerateCannyImage] - Canny image generation complete.");
                    return new InputImage(maskImage);
                }
            }
        }


        /// <summary>
        /// Generates the hard edge image mask.
        /// </summary>
        /// <param name="controlNetModel">The control net model.</param>
        /// <param name="inputImage">The input image.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <returns></returns>
        private async Task<InputImage> GenerateHardEdgeImage(ControlNetModelSet controlNetModel, InputImage inputImage, int height, int width)
        {
            _logger.LogInformation($"[GenerateHardEdgeImage] - Generating HardEdge image...");
            var controlImage = inputImage.ToDenseTensor(new[] { 1, 3, height, width }, false);
            var metadata = _onnxModelService.GetModelMetadata(controlNetModel, OnnxModelType.Annotation);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(controlImage);
                inferenceParameters.AddOutputBuffer(new[] { 1, 1, height, width });

                var results = await _onnxModelService.RunInferenceAsync(controlNetModel, OnnxModelType.Annotation, inferenceParameters);
                using (var result = results.First())
                {
                    var testImage = result.ToDenseTensor().Repeat(3);
                    var imageTensor = new DenseTensor<float>(controlImage.Dimensions);
                    for (int i = 0; i < testImage.Length; i++)
                        imageTensor.SetValue(i, testImage.GetValue(i));

                    var maskImage = imageTensor.ToImageMask();
                    //await maskImage.SaveAsPngAsync("D:\\Hed.png");
                    _logger.LogInformation($"[GenerateHardEdgeImage] - HardEdge image generation complete.");
                    return new InputImage(maskImage);
                }
            }
        }


        /// <summary>
        /// Generates the depth image mask.
        /// </summary>
        /// <param name="controlNetModel">The control net model.</param>
        /// <param name="inputImage">The input image.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <returns></returns>
        private async Task<InputImage> GenerateDepthImage(ControlNetModelSet controlNetModel, InputImage inputImage, int height, int width)
        {
            _logger.LogInformation($"[GenerateDepthImage] - Generating Depth image...");
            var controlImage = inputImage.ToDenseTensor(new[] { 1, 3, height, width }, false);
            var metadata = _onnxModelService.GetModelMetadata(controlNetModel, OnnxModelType.Annotation);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(controlImage);
                inferenceParameters.AddOutputBuffer(new[] { 1, 1, height, width });

                var results = await _onnxModelService.RunInferenceAsync(controlNetModel, OnnxModelType.Annotation, inferenceParameters);
                using (var result = results.First())
                {
                    var imageResult = result.ToDenseTensor();
                    var imageTensor = new DenseTensor<float>(controlImage.Dimensions);
                    for (int i = 0; i < imageResult.Length; i++)
                    {
                        imageTensor.SetValue(i, imageResult.GetValue(i));
                    }

                    NormalizeDepthTensor(imageTensor);
                    var maskImage = imageTensor.ToImageMask();
                    //await maskImage.SaveAsPngAsync("D:\\Depth.png");
                    _logger.LogInformation($"[GenerateDepthImage] - Depth image generation complete.");
                    return new InputImage(maskImage);
                }
            }
        }


        /// <summary>
        /// Prepares the input image.
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <returns></returns>
        private static InputImage PrepareInputImage(InputImage inputImage, int height, int width)
        {
            return inputImage;
        }


        /// <summary>
        /// Normalizes the depth tensor.
        /// </summary>
        /// <param name="value">The value.</param>
        public static void NormalizeDepthTensor(DenseTensor<float> value)
        {
            var values = value.Buffer.Span;
            float min = float.PositiveInfinity, max = float.NegativeInfinity;
            foreach (var val in values)
            {
                if (min > val) min = val;
                if (max < val) max = val;
            }

            var range = max - min;
            for (var i = 0; i < values.Length; i++)
            {
                values[i] = (values[i] - min) / range;
            }
        }
    }
}
