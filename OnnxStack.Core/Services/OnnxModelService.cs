using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace OnnxStack.Core.Services
{
    /// <summary>
    /// Service to cache the ONNX model instances for faster inference
    /// </summary>
    /// <seealso cref="OnnxStack.Core.Services.IOnnxModelService" />
    public sealed class OnnxModelService : IOnnxModelService
    {
        private readonly OnnxStackConfig _configuration;
        private readonly ConcurrentDictionary<string, OnnxModelSet> _onnxModelSets;
        private readonly ConcurrentDictionary<string, OnnxModelSetConfig> _onnxModelSetConfigs;

        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxModelService"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public OnnxModelService(OnnxStackConfig configuration)
        {
            _configuration = configuration;
            _onnxModelSets = new ConcurrentDictionary<string, OnnxModelSet>();
            _onnxModelSetConfigs = _configuration.OnnxModelSets.ToConcurrentDictionary(x => x.Name, x => x);
        }


        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        public async Task<OnnxModelSet> LoadModel(IOnnxModel model)
        {
            return await Task.Run(() => LoadModelSet(model)).ConfigureAwait(false);
        }

        /// <summary>
        /// Unloads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        public async Task<bool> UnloadModel(IOnnxModel model)
        {
            return await Task.Run(() => UnloadModelSet(model)).ConfigureAwait(false);
        }


        /// <summary>
        /// Determines whether the specified model is loaded.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns>
        ///   <c>true</c> if the specified model is loaded; otherwise, <c>false</c>.
        /// </returns>
        public bool IsModelLoaded(IOnnxModel model)
        {
            return _onnxModelSets.ContainsKey(model.Name);
        }


        /// <summary>
        /// Determines whether the specified model type is enabled.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <returns>
        ///   <c>true</c> if the specified model type is enabled; otherwise, <c>false</c>.
        /// </returns>
        public bool IsEnabled(IOnnxModel model, OnnxModelType modelType)
        {
            return GetModelSet(model).Exists(modelType);
        }


        /// <summary>
        /// Determines whether the specified model type is enabled.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <returns>
        ///   <c>true</c> if the specified model type is enabled; otherwise, <c>false</c>.
        /// </returns>
        public Task<bool> IsEnabledAsync(IOnnxModel model, OnnxModelType modelType)
        {
            return Task.FromResult(IsEnabled(model, modelType));
        }


        /// <summary>
        /// Runs inference on the specified model.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="inputs">The inputs.</param>
        /// <returns></returns>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunInference(IOnnxModel model, OnnxModelType modelType, IReadOnlyCollection<NamedOnnxValue> inputs)
        {
            return RunInternal(model, modelType, inputs);
        }


        /// <summary>
        /// Runs inference on the specified model asynchronously(ish).
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="inputs">The inputs.</param>
        /// <returns></returns>
        public async Task<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> RunInferenceAsync(IOnnxModel model, OnnxModelType modelType, IReadOnlyCollection<NamedOnnxValue> inputs)
        {
            return await Task.Run(() => RunInternal(model, modelType, inputs)).ConfigureAwait(false);
        }


        /// <summary>
        /// Gets the input metadata.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <returns></returns>
        /// <exception cref="System.NotImplementedException"></exception>
        public IReadOnlyDictionary<string, NodeMetadata> GetInputMetadata(IOnnxModel model, OnnxModelType modelType)
        {
            return InputMetadataInternal(model, modelType);
        }


        /// <summary>
        /// Gets the input names.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <returns></returns>
        /// <exception cref="System.NotImplementedException"></exception>
        public IReadOnlyList<string> GetInputNames(IOnnxModel model, OnnxModelType modelType)
        {
            return InputNamesInternal(model, modelType);
        }


        /// <summary>
        /// Gets the output metadata.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <returns></returns>
        /// <exception cref="System.NotImplementedException"></exception>
        public IReadOnlyDictionary<string, NodeMetadata> GetOutputMetadata(IOnnxModel model, OnnxModelType modelType)
        {
            return OutputMetadataInternal(model, modelType);
        }


        /// <summary>
        /// Gets the output names.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <returns></returns>
        /// <exception cref="System.NotImplementedException"></exception>
        public IReadOnlyList<string> GetOutputNames(IOnnxModel model, OnnxModelType modelType)
        {
            return OutputNamesInternal(model, modelType);
        }


        /// <summary>
        /// Runs inference on the specified model.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="inputs">The inputs.</param>
        /// <returns></returns>
        private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunInternal(IOnnxModel model, OnnxModelType modelType, IReadOnlyCollection<NamedOnnxValue> inputs)
        {
            return GetModelSet(model)
                .GetSession(modelType)
                .Run(inputs);
        }



        /// <summary>
        /// Gets the Sessions input metadata.
        /// </summary>
        /// <param name="modelType">Type of model.</param>
        /// <returns></returns>
        private IReadOnlyDictionary<string, NodeMetadata> InputMetadataInternal(IOnnxModel model, OnnxModelType modelType)
        {
            return GetModelSet(model)
                .GetSession(modelType)
                .InputMetadata;
        }

        /// <summary>
        /// Gets the Sessions input names.
        /// </summary>
        /// <param name="modelType">Type of model.</param>
        /// <returns></returns>
        private IReadOnlyList<string> InputNamesInternal(IOnnxModel model, OnnxModelType modelType)
        {
            return GetModelSet(model)
                .GetSession(modelType)
                .InputNames;
        }

        /// <summary>
        /// Gets the Sessions output metadata.
        /// </summary>
        /// <param name="modelType">Type of model.</param>
        /// <returns></returns>
        private IReadOnlyDictionary<string, NodeMetadata> OutputMetadataInternal(IOnnxModel model, OnnxModelType modelType)
        {
            return GetModelSet(model)
                .GetSession(modelType)
                .OutputMetadata;
        }

        /// <summary>
        /// Gets the Sessions output metadata names.
        /// </summary>
        /// <param name="modelType">Type of model.</param>
        /// <returns></returns>
        private IReadOnlyList<string> OutputNamesInternal(IOnnxModel model, OnnxModelType modelType)
        {
            return GetModelSet(model)
                .GetSession(modelType)
                .OutputNames;
        }


        /// <summary>
        /// Gets the model set.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        /// <exception cref="System.Exception">Model {model.Name} has not been loaded</exception>
        private OnnxModelSet GetModelSet(IOnnxModel model)
        {
            if (!_onnxModelSets.TryGetValue(model.Name, out var modelSet))
                throw new Exception($"Model {model.Name} has not been loaded");

            return modelSet;
        }


        /// <summary>
        /// Loads the model set.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        /// <exception cref="System.Exception">Model {model.Name} not found in configuration</exception>
        private OnnxModelSet LoadModelSet(IOnnxModel model)
        {
            if (_onnxModelSets.ContainsKey(model.Name))
                return _onnxModelSets[model.Name];

            if (!_onnxModelSetConfigs.TryGetValue(model.Name, out var modelSetConfig))
                throw new Exception($"Model {model.Name} not found in configuration");

            if (!modelSetConfig.IsEnabled)
                throw new Exception($"Model {model.Name} is not enabled");

            var modelSet = new OnnxModelSet(modelSetConfig);
            _onnxModelSets.TryAdd(model.Name, modelSet);
            return modelSet;
        }


        /// <summary>
        /// Unloads the model set.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        private bool UnloadModelSet(IOnnxModel model)
        {
            if (!_onnxModelSets.TryGetValue(model.Name, out var modelSet))
                return true;

            if (_onnxModelSets.TryRemove(model.Name, out modelSet))
            {
                modelSet?.Dispose();
                return true;
            }
            return false;
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            foreach (var onnxModelSet in _onnxModelSets.Values)
            {
                onnxModelSet?.Dispose();
            }
        }
    }
}
