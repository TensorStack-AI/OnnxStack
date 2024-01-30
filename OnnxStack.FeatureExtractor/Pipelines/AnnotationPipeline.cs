using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.FeatureExtractor.Common;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace OnnxStack.FeatureExtractor.Pipelines
{
    public class AnnotationPipeline
    {
        private readonly string _name;
        private readonly ILogger _logger;
        private readonly FeatureExtractorModel _cannyModel;
        private readonly FeatureExtractorModel _hedModel;
        private readonly FeatureExtractorModel _depthModel;

        /// <summary>
        /// Initializes a new instance of the <see cref="AnnotationPipeline"/> class.
        /// </summary>
        /// <param name="name">The name.</param>
        /// <param name="cannyModel">The canny model.</param>
        /// <param name="hedModel">The hed model.</param>
        /// <param name="depthModel">The depth model.</param>
        /// <param name="logger">The logger.</param>
        public AnnotationPipeline(string name, FeatureExtractorModel cannyModel, FeatureExtractorModel hedModel, FeatureExtractorModel depthModel, ILogger logger = default)
        {
            _name = name;
            _logger = logger;
            _cannyModel = cannyModel;
            _hedModel = hedModel;
            _depthModel = depthModel;
        }


        /// <summary>
        /// Gets the name.
        /// </summary>
        /// <value>
        public string Name => _name;


        /// <summary>
        /// Unloads the models.
        /// </summary>
        public async Task UnloadAsync()
        {
            await Task.Yield();
            _cannyModel?.Dispose();
            _hedModel?.Dispose();
            _depthModel?.Dispose();
        }


        /// <summary>
        /// Generates the Canny image mask.
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<InputImage> CannyImage(InputImage inputImage)
        {
            var timestamp = _logger?.LogBegin("Generating Canny image...");
            var controlImage = await inputImage.ToDenseTensorAsync(_depthModel.SampleSize, _depthModel.SampleSize, ImageNormalizeType.ZeroToOne);
            var metadata = await _cannyModel.GetMetadataAsync();
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(controlImage);
                inferenceParameters.AddOutputBuffer(new[] { 1, 1, _depthModel.SampleSize, _depthModel.SampleSize });

                var results = await _cannyModel.RunInferenceAsync(inferenceParameters);
                using (var result = results.First())
                {
                    var testImage = result.ToDenseTensor().Repeat(3);
                    var imageTensor = new DenseTensor<float>(controlImage.Dimensions);
                    for (int i = 0; i < testImage.Length; i++)
                        imageTensor.SetValue(i, testImage.GetValue(i));

                    var maskImage = imageTensor.ToImageMask();
                    //await maskImage.SaveAsPngAsync("D:\\Canny.png");
                    _logger?.LogEnd("Generating Canny image complete", timestamp);
                    return new InputImage(maskImage);
                }
            }
        }


        /// <summary>
        /// Generates the HED image mask.
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public async Task<InputImage> HedImage(InputImage inputImage)
        {
            var timestamp = _logger?.LogBegin("Generating Hed image...");
            var controlImage = await inputImage.ToDenseTensorAsync(_depthModel.SampleSize, _depthModel.SampleSize, ImageNormalizeType.ZeroToOne);
            var metadata = await _hedModel.GetMetadataAsync();
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(controlImage);
                inferenceParameters.AddOutputBuffer(new[] { 1, 1, _depthModel.SampleSize, _depthModel.SampleSize });

                var results = await _hedModel.RunInferenceAsync(inferenceParameters);
                using (var result = results.First())
                {
                    var testImage = result.ToDenseTensor().Repeat(3);
                    var imageTensor = new DenseTensor<float>(controlImage.Dimensions);
                    for (int i = 0; i < testImage.Length; i++)
                        imageTensor.SetValue(i, testImage.GetValue(i));

                    var maskImage = imageTensor.ToImageMask();
                    //await maskImage.SaveAsPngAsync("D:\\Hed.png");
                    _logger?.LogEnd("Generating Hed image complete", timestamp);
                    return new InputImage(maskImage);
                }
            }
        }


        /// <summary>
        /// Generates the depth image mask.
        /// </summary>
        /// <param name="modelSet">The control net model.</param>
        /// <param name="inputImage">The input image.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <returns></returns>
        public async Task<InputImage> DepthImage(InputImage inputImage)
        {
            var timestamp = _logger?.LogBegin("Generating Depth image...");
            var controlImage = await inputImage.ToDenseTensorAsync(_depthModel.SampleSize, _depthModel.SampleSize, ImageNormalizeType.ZeroToOne);
            var metadata = await _depthModel.GetMetadataAsync();
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(controlImage);
                inferenceParameters.AddOutputBuffer(new[] { 1, 1, _depthModel.SampleSize, _depthModel.SampleSize });

                var results = await _depthModel.RunInferenceAsync(inferenceParameters);
                using (var result = results.First())
                {
                    var testImage = result.ToDenseTensor().Repeat(3);
                    var imageTensor = new DenseTensor<float>(controlImage.Dimensions);
                    for (int i = 0; i < testImage.Length; i++)
                        imageTensor.SetValue(i, testImage.GetValue(i));

                    NormalizeDepthTensor(imageTensor);
                    var maskImage = imageTensor.ToImageMask();
                    //await maskImage.SaveAsPngAsync("D:\\Depth.png");
                    _logger?.LogEnd("Generating Depth image complete", timestamp);
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


        /// <summary>
        /// Creates the pipeline from a AnnotationModelSet.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static AnnotationPipeline CreatePipeline(AnnotationModelSet modelSet, ILogger logger = default)
        {
            var canny = new FeatureExtractorModel(modelSet.CannyImageConfig.ApplyDefaults(modelSet));
            var hed = new FeatureExtractorModel(modelSet.HedImageConfig.ApplyDefaults(modelSet));
            var depth = new FeatureExtractorModel(modelSet.DepthImageConfig.ApplyDefaults(modelSet));
            return new AnnotationPipeline(modelSet.Name, canny, hed, depth, logger);
        }


        /// <summary>
        /// Creates the pipeline from the specified folder.
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static AnnotationPipeline CreatePipeline(string modelFolder, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML, ILogger logger = default)
        {
            var name = Path.GetFileNameWithoutExtension(modelFolder);
            var canny = Path.Combine(modelFolder, "canny.onnx");
            var hed = Path.Combine(modelFolder, "hed.onnx");
            var depth = Path.Combine(modelFolder, "depth.onnx");
            var configuration = new AnnotationModelSet
            {
                Name = name,
                IsEnabled = true,
                DeviceId = deviceId,
                ExecutionProvider = executionProvider,
                CannyImageConfig = new FeatureExtractorModelConfig { OnnxModelPath = canny, SampleSize = 512 },
                HedImageConfig = new FeatureExtractorModelConfig { OnnxModelPath = hed, SampleSize = 512 },
                DepthImageConfig = new FeatureExtractorModelConfig { OnnxModelPath = depth, SampleSize = 512 }
            };
            return CreatePipeline(configuration, logger);
        }
    }
}
