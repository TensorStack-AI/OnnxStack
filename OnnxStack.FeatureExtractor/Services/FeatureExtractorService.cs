using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.FeatureExtractor.Common;
using System.Collections.Generic;
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
        private readonly Dictionary<IOnnxModel, OnnxModelSession> _modelSessions;


        /// <summary>
        /// Initializes a new instance of the <see cref="FeatureExtractorService"/> class.
        /// </summary>
        /// <param name="onnxModelService">The onnx model service.</param>
        public FeatureExtractorService(ILogger<FeatureExtractorService> logger = default)
        {
            _logger = logger;
            _modelSessions = new Dictionary<IOnnxModel, OnnxModelSession>();
        }


        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        public Task<bool> LoadModelAsync(FeatureExtractorModelSet model)
        {
            if (_modelSessions.ContainsKey(model))
                return Task.FromResult(true);

            return Task.FromResult(_modelSessions.TryAdd(model, CreateModelSession(model)));
        }


        /// <summary>
        /// Unloads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        public Task<bool> UnloadModelAsync(FeatureExtractorModelSet model)
        {
            if (_modelSessions.Remove(model, out var session))
            {
                session?.Dispose();
            }
            return Task.FromResult(true);
        }


        /// <summary>
        /// Determines whether [is model loaded] [the specified model options].
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <returns>
        ///   <c>true</c> if [is model loaded] [the specified model options]; otherwise, <c>false</c>.
        /// </returns>
        /// <exception cref="System.NotImplementedException"></exception>
        public bool IsModelLoaded(FeatureExtractorModelSet modelOptions)
        {
            return _modelSessions.ContainsKey(modelOptions);
        }

        /// <summary>
        /// Generates the canny image mask.
        /// </summary>
        /// <param name="modelSet">The control net model.</param>
        /// <param name="inputImage">The input image.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <returns></returns>
        public async Task<InputImage> CannyImage(FeatureExtractorModelSet modelSet, InputImage inputImage, int height, int width)
        {
            _logger?.LogInformation($"[CannyImage] - Generating Canny image...");
            if (!_modelSessions.TryGetValue(modelSet, out var modelSession))
                throw new System.Exception("Model not loaded");

            var controlImage = await inputImage.ToDenseTensorAsync(height, width, ImageNormalizeType.ZeroToOne);
            var metadata = await modelSession.GetMetadataAsync();
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(controlImage);
                inferenceParameters.AddOutputBuffer(new[] { 1, 1, height, width });

                var results = await modelSession.RunInferenceAsync(inferenceParameters);
                using (var result = results.First())
                {
                    var testImage = result.ToDenseTensor().Repeat(3);
                    var imageTensor = new DenseTensor<float>(controlImage.Dimensions);
                    for (int i = 0; i < testImage.Length; i++)
                        imageTensor.SetValue(i, testImage.GetValue(i));

                    var maskImage = imageTensor.ToImageMask();
                    //await maskImage.SaveAsPngAsync("D:\\Canny.png");
                    _logger?.LogInformation($"[CannyImage] - Canny image generation complete.");
                    return new InputImage(maskImage);
                }
            }
        }


        /// <summary>
        /// Generates the hard edge image mask.
        /// </summary>
        /// <param name="modelSet">The control net model.</param>
        /// <param name="inputImage">The input image.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <returns></returns>
        public async Task<InputImage> HedImage(FeatureExtractorModelSet modelSet, InputImage inputImage, int height, int width)
        {
            _logger?.LogInformation($"[HedImage] - Generating HardEdge image...");
            if (!_modelSessions.TryGetValue(modelSet, out var modelSession))
                throw new System.Exception("Model not loaded");

            var controlImage = await inputImage.ToDenseTensorAsync(height, width, ImageNormalizeType.ZeroToOne);
            var metadata = await modelSession.GetMetadataAsync();
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(controlImage);
                inferenceParameters.AddOutputBuffer(new[] { 1, 1, height, width });

                var results = await modelSession.RunInferenceAsync(inferenceParameters);
                using (var result = results.First())
                {
                    var testImage = result.ToDenseTensor().Repeat(3);
                    var imageTensor = new DenseTensor<float>(controlImage.Dimensions);
                    for (int i = 0; i < testImage.Length; i++)
                        imageTensor.SetValue(i, testImage.GetValue(i));

                    var maskImage = imageTensor.ToImageMask();
                    //await maskImage.SaveAsPngAsync("D:\\Hed.png");
                    _logger?.LogInformation($"[HedImage] - HardEdge image generation complete.");
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
        public async Task<InputImage> DepthImage(FeatureExtractorModelSet modelSet, InputImage inputImage, int height, int width)
        {
            _logger?.LogInformation($"[DepthImage] - Generating Depth image...");
            if (!_modelSessions.TryGetValue(modelSet, out var modelSession))
                throw new System.Exception("Model not loaded");

            var controlImage = await inputImage.ToDenseTensorAsync(height, width, ImageNormalizeType.ZeroToOne);
            var metadata = await modelSession.GetMetadataAsync();
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(controlImage);
                inferenceParameters.AddOutputBuffer(new[] { 1, 1, height, width });

                var results = await modelSession.RunInferenceAsync(inferenceParameters);
                using (var result = results.First())
                {
                    var testImage = result.ToDenseTensor().Repeat(3);
                    var imageTensor = new DenseTensor<float>(controlImage.Dimensions);
                    for (int i = 0; i < testImage.Length; i++)
                        imageTensor.SetValue(i, testImage.GetValue(i));

                    NormalizeDepthTensor(imageTensor);
                    var maskImage = imageTensor.ToImageMask();
                    //await maskImage.SaveAsPngAsync("D:\\Depth.png");
                    _logger?.LogInformation($"[DepthImage] - Depth image generation complete.");
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

        private OnnxModelSession CreateModelSession(FeatureExtractorModelSet modelSet)
        {
            modelSet.ModelConfigurations.ForEach(x => x.ApplyDefaults(modelSet));
            return new OnnxModelSession(modelSet.ModelConfigurations.FirstOrDefault());
        }
    }
}
