using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace OnnxStack.Core.Services
{
    /// <summary>
    /// Service to cache the ONNX model instances for faster inference
    /// </summary>
    /// <seealso cref="OnnxStack.Core.Services.IOnnxModelService" />
    public class OnnxModelService : IOnnxModelService
    {
        private readonly SessionOptions _sessionOptions;
        private readonly OnnxStackConfig _configuration;
        private readonly InferenceSession _onnxUnetInferenceSession;
        private readonly InferenceSession _onnxTokenizerInferenceSession;
        private readonly InferenceSession _onnxVaeDecoderInferenceSession;
        private readonly InferenceSession _onnxTextEncoderInferenceSession;
        private readonly InferenceSession _onnxSafetyModelInferenceSession;

        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxModelService"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public OnnxModelService(OnnxStackConfig configuration)
        {
            _configuration = configuration;
            _sessionOptions = _configuration.GetSessionOptions();
            _sessionOptions.RegisterOrtExtensions();
            _onnxUnetInferenceSession = new InferenceSession(_configuration.OnnxUnetPath, _sessionOptions);
            _onnxTokenizerInferenceSession = new InferenceSession(_configuration.OnnxTokenizerPath, _sessionOptions);
            _onnxVaeDecoderInferenceSession = new InferenceSession(_configuration.OnnxVaeDecoderPath, _sessionOptions);
            _onnxTextEncoderInferenceSession = new InferenceSession(_configuration.OnnxTextEncoderPath, _sessionOptions);
            if (_configuration.IsSafetyModelEnabled)
                _onnxSafetyModelInferenceSession = new InferenceSession(_configuration.OnnxSafetyModelPath, _sessionOptions);
        }


        /// <summary>
        /// Runs inference on the specified model.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="inputs">The inputs.</param>
        /// <returns></returns>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunInference(OnnxModelType modelType, IReadOnlyCollection<NamedOnnxValue> inputs)
        {
            return RunInternal(modelType, inputs);
        }


        /// <summary>
        /// Runs inference on the specified model asynchronously(ish).
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="inputs">The inputs.</param>
        /// <returns></returns>
        public async Task<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> RunInferenceAsync(OnnxModelType modelType, IReadOnlyCollection<NamedOnnxValue> inputs)
        {
            return await Task.Run(() => RunInternal(modelType, inputs)).ConfigureAwait(false);
        }

        /// <summary>
        /// Runs inference on the specified model.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="inputs">The inputs.</param>
        /// <returns></returns>
        private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunInternal(OnnxModelType modelType, IReadOnlyCollection<NamedOnnxValue> inputs)
        {
            return modelType switch
            {
                OnnxModelType.Unet => _onnxUnetInferenceSession.Run(inputs),
                OnnxModelType.Tokenizer => _onnxTokenizerInferenceSession.Run(inputs),
                OnnxModelType.VaeDecoder => _onnxVaeDecoderInferenceSession.Run(inputs),
                OnnxModelType.TextEncoder => _onnxTextEncoderInferenceSession.Run(inputs),
                OnnxModelType.SafetyModel => _onnxSafetyModelInferenceSession.Run(inputs),
                _ => default
            };
        }


        public void Dispose()
        {
            _sessionOptions?.Dispose();
            _onnxUnetInferenceSession?.Dispose();
            _onnxTokenizerInferenceSession?.Dispose();
            _onnxVaeDecoderInferenceSession?.Dispose();
            _onnxTextEncoderInferenceSession?.Dispose();
            _onnxSafetyModelInferenceSession?.Dispose();
        }
    }
}
