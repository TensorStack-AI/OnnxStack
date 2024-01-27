using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.Core.Services;
using OnnxStack.FeatureExtractor.Common;
using System.Linq;
using System.Threading.Tasks;

namespace OnnxStack.FeatureExtractor.Services
{
    /// <summary>
    /// Service for handing images for input and output of the diffusion process
    /// </summary>
    /// <seealso cref="OnnxStack.StableDiffusion.Common.IFeatureExtractor" />
    public class FeatureExtractorService : IFeatureExtractorService
    {
        private readonly ILogger<FeatureExtractorService> _logger;
        private readonly IOnnxModelService _onnxModelService;


        /// <summary>
        /// Initializes a new instance of the <see cref="FeatureExtractorService"/> class.
        /// </summary>
        /// <param name="onnxModelService">The onnx model service.</param>
        public FeatureExtractorService(IOnnxModelService onnxModelService, ILogger<FeatureExtractorService> logger)
        {
            _logger = logger;
            _onnxModelService = onnxModelService;
        }


        /// <summary>
        /// Generates the canny image mask.
        /// </summary>
        /// <param name="controlNetModel">The control net model.</param>
        /// <param name="inputImage">The input image.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <returns></returns>
        public async Task<InputImage> CannyImage(FeatureExtractorModelSet controlNetModel, InputImage inputImage, int height, int width)
        {
            _logger.LogInformation($"[CannyImage] - Generating Canny image...");
            var controlImage = await inputImage.ToDenseTensorAsync(height, width, ImageNormalizeType.ZeroToOne);
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
                    _logger.LogInformation($"[CannyImage] - Canny image generation complete.");
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
        public async Task<InputImage> HedImage(FeatureExtractorModelSet controlNetModel, InputImage inputImage, int height, int width)
        {
            _logger.LogInformation($"[HedImage] - Generating HardEdge image...");
            var controlImage = await inputImage.ToDenseTensorAsync(height, width, ImageNormalizeType.ZeroToOne);
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
                    _logger.LogInformation($"[HedImage] - HardEdge image generation complete.");
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
        public async Task<InputImage> DepthImage(FeatureExtractorModelSet controlNetModel, InputImage inputImage, int height, int width)
        {
            _logger.LogInformation($"[DepthImage] - Generating Depth image...");
            var controlImage = await inputImage.ToDenseTensorAsync(height, width, ImageNormalizeType.ZeroToOne);
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

                    NormalizeDepthTensor(imageTensor);
                    var maskImage = imageTensor.ToImageMask();
                    //await maskImage.SaveAsPngAsync("D:\\Depth.png");
                    _logger.LogInformation($"[DepthImage] - Depth image generation complete.");
                    return new InputImage(maskImage);
                }
            }
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
