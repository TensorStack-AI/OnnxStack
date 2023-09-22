using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using System.Collections.Generic;
using System.Collections.Immutable;
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
        private readonly ImmutableDictionary<OnnxModelType, InferenceSession> _modelInferenceSessions;

        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxModelService"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public OnnxModelService(OnnxStackConfig configuration)
        {
            _configuration = configuration;
            _sessionOptions = _configuration.GetSessionOptions();
            _sessionOptions.RegisterOrtExtensions();
            var modelInferenceSessions = new Dictionary<OnnxModelType, InferenceSession>
            {
                {OnnxModelType.Unet,new InferenceSession(_configuration.OnnxUnetPath, _sessionOptions) },
                {OnnxModelType.Tokenizer,new InferenceSession(_configuration.OnnxTokenizerPath, _sessionOptions) },
                {OnnxModelType.VaeDecoder,new InferenceSession(_configuration.OnnxVaeDecoderPath, _sessionOptions) },
                {OnnxModelType.TextEncoder,new InferenceSession(_configuration.OnnxTextEncoderPath, _sessionOptions)}
            };
            if (_configuration.IsSafetyModelEnabled)
                modelInferenceSessions.Add(OnnxModelType.SafetyModel, new InferenceSession(_configuration.OnnxSafetyModelPath, _sessionOptions));

            _modelInferenceSessions = modelInferenceSessions.ToImmutableDictionary();
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
            return _modelInferenceSessions[modelType]?.Run(inputs);
        }


        public void Dispose()
        {
            _sessionOptions?.Dispose();
            foreach (var modelInferenceSession in _modelInferenceSessions.Values)
            {
                modelInferenceSession?.Dispose();
            }
        }
    }
}
