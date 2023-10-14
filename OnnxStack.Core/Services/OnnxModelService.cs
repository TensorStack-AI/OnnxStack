using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace OnnxStack.Core.Services
{
    /// <summary>
    /// Service to cache the ONNX model instances for faster inference
    /// </summary>
    /// <seealso cref="OnnxStack.Core.Services.IOnnxModelService" />
    public sealed class OnnxModelService : IOnnxModelService
    {
        private readonly OnnxModelSet _onnxModelSet;
        private readonly OnnxStackConfig _configuration;
        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxModelService"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public OnnxModelService(OnnxStackConfig configuration)
        {
            _configuration = configuration;
            _onnxModelSet = new OnnxModelSet(configuration);
        }


        /// <summary>
        /// Gets the configuration.
        /// </summary>
        public OnnxStackConfig Configuration => _configuration;


        /// <summary>
        /// Determines whether the specified model type is enabled.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <returns>
        ///   <c>true</c> if the specified model type is enabled; otherwise, <c>false</c>.
        /// </returns>
        public bool IsEnabled(OnnxModelType modelType)
        {
            return _onnxModelSet.Exists(modelType);
        }


        /// <summary>
        /// Determines whether the specified model type is enabled.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <returns>
        ///   <c>true</c> if the specified model type is enabled; otherwise, <c>false</c>.
        /// </returns>
        public Task<bool> IsEnabledAsync(OnnxModelType modelType)
        {
            return Task.FromResult(IsEnabled(modelType));
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
        /// Gets the input metadata.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <returns></returns>
        /// <exception cref="System.NotImplementedException"></exception>
        public IReadOnlyDictionary<string, NodeMetadata> GetInputMetadata(OnnxModelType modelType)
        {
            return InputMetadataInternal(modelType);
        }


        /// <summary>
        /// Gets the input names.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <returns></returns>
        /// <exception cref="System.NotImplementedException"></exception>
        public IReadOnlyList<string> GetInputNames(OnnxModelType modelType)
        {
            return InputNamesInternal(modelType);
        }


        /// <summary>
        /// Gets the output metadata.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <returns></returns>
        /// <exception cref="System.NotImplementedException"></exception>
        public IReadOnlyDictionary<string, NodeMetadata> GetOutputMetadata(OnnxModelType modelType)
        {
           return OutputMetadataInternal(modelType);
        }


        /// <summary>
        /// Gets the output names.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <returns></returns>
        /// <exception cref="System.NotImplementedException"></exception>
        public IReadOnlyList<string> GetOutputNames(OnnxModelType modelType)
        {
            return OutputNamesInternal(modelType);
        }


        /// <summary>
        /// Runs inference on the specified model.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="inputs">The inputs.</param>
        /// <returns></returns>
        private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunInternal(OnnxModelType modelType, IReadOnlyCollection<NamedOnnxValue> inputs)
        {
            return _onnxModelSet.GetSession(modelType).Run(inputs);
        }



        /// <summary>
        /// Gets the Sessions input metadata.
        /// </summary>
        /// <param name="modelType">Type of model.</param>
        /// <returns></returns>
        private IReadOnlyDictionary<string, NodeMetadata> InputMetadataInternal(OnnxModelType modelType)
        {
            return _onnxModelSet.GetSession(modelType).InputMetadata;
        }

        /// <summary>
        /// Gets the Sessions input names.
        /// </summary>
        /// <param name="modelType">Type of model.</param>
        /// <returns></returns>
        private IReadOnlyList<string> InputNamesInternal(OnnxModelType modelType)
        {
            return _onnxModelSet.GetSession(modelType).InputNames;
        }

        /// <summary>
        /// Gets the Sessions output metadata.
        /// </summary>
        /// <param name="modelType">Type of model.</param>
        /// <returns></returns>
        private IReadOnlyDictionary<string, NodeMetadata> OutputMetadataInternal(OnnxModelType modelType)
        {
            return _onnxModelSet.GetSession(modelType).OutputMetadata;
        }

        /// <summary>
        /// Gets the Sessions output metadata names.
        /// </summary>
        /// <param name="modelType">Type of model.</param>
        /// <returns></returns>
        private IReadOnlyList<string> OutputNamesInternal(OnnxModelType modelType)
        {
            return _onnxModelSet.GetSession(modelType).OutputNames;
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            _onnxModelSet?.Dispose();
        }
    }
}
