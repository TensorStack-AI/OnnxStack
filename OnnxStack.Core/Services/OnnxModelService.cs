﻿using Microsoft.ML.OnnxRuntime;
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
        private readonly ConcurrentDictionary<IOnnxModel, OnnxModelSet> _onnxModelSets;

        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxModelService"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public OnnxModelService(OnnxStackConfig configuration)
        {
            _configuration = configuration;
            _onnxModelSets = new ConcurrentDictionary<IOnnxModel, OnnxModelSet>(new OnnxModelEqualityComparer());
        }


        /// <summary>
        /// Gets the active loaded ModelSets.
        /// </summary>
        public IEnumerable<OnnxModelSet> ModelSets => _onnxModelSets.Values;


        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        public async Task<OnnxModelSet> LoadModelAsync(IOnnxModelSetConfig model)
        {
            return await Task.Run(() => LoadModelSet(model)).ConfigureAwait(false);
        }

        /// <summary>
        /// Unloads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        public async Task<bool> UnloadModelAsync(IOnnxModel model)
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
            return _onnxModelSets.ContainsKey(model);
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
        /// Runs the inference (Use when output size is unknown)
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="inputName">Name of the input.</param>
        /// <param name="inputValue">The input value.</param>
        /// <param name="outputName">Name of the output.</param>
        /// <returns></returns>
        public IDisposableReadOnlyCollection<OrtValue> RunInference(IOnnxModel model, OnnxModelType modelType, OnnxInferenceParameters parameters)
        {
            return RunInferenceInternal(model, modelType, parameters);
        }


        /// <summary>
        /// Runs the inference asynchronously, (Use when output size is known)
        /// Output buffer size must be known and set before inference is run
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="inputs">The inputs.</param>
        /// <param name="outputs">The outputs.</param>
        /// <returns></returns>
        public Task<IReadOnlyCollection<OrtValue>> RunInferenceAsync(IOnnxModel model, OnnxModelType modelType, OnnxInferenceParameters parameters)
        {
            return RunInferenceInternalAsync(model, modelType, parameters);
        }


        /// <summary>
        /// Gets the model metadata.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <returns></returns>
        public OnnxMetadata GetModelMetadata(IOnnxModel model, OnnxModelType modelType)
        {
            return GetNodeMetadataInternal(model, modelType);
        }


        /// <summary>
        /// Runs the inference.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="parameters">The parameters.</param>
        /// <returns></returns>
        private IDisposableReadOnlyCollection<OrtValue> RunInferenceInternal(IOnnxModel model, OnnxModelType modelType, OnnxInferenceParameters parameters)
        {
            return GetModelSet(model)
                .GetSession(modelType)
                .Run(parameters.RunOptions, parameters.InputNameValues, parameters.OutputNames);
        }


        /// <summary>
        /// Runs the inference asynchronously.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="parameters">The parameters.</param>
        /// <returns></returns>
        private Task<IReadOnlyCollection<OrtValue>> RunInferenceInternalAsync(IOnnxModel model, OnnxModelType modelType, OnnxInferenceParameters parameters)
        {
            return GetModelSet(model)
                .GetSession(modelType)
                .RunAsync(parameters.RunOptions, parameters.InputNames, parameters.InputValues, parameters.OutputNames, parameters.OutputValues);
        }


        /// <summary>
        /// Gets the node metadata.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <returns></returns>
        private OnnxMetadata GetNodeMetadataInternal(IOnnxModel model, OnnxModelType modelType)
        {
            var session = GetModelSet(model).GetSession(modelType);
            return new OnnxMetadata
            {
                Inputs = session.InputMetadata
                    .Select(OnnxNamedMetadata.Create)
                    .ToList(),
                Outputs = session.OutputMetadata
                    .Select(OnnxNamedMetadata.Create)
                    .ToList()
            };
        }


        /// <summary>
        /// Gets the model set.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        /// <exception cref="System.Exception">Model {model.Name} has not been loaded</exception>
        private OnnxModelSet GetModelSet(IOnnxModel model)
        {
            if (!_onnxModelSets.TryGetValue(model, out var modelSet))
                throw new Exception($"Model {model.Name} has not been loaded");

            return modelSet;
        }


        /// <summary>
        /// Loads the model set.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        /// <exception cref="System.Exception">Model {model.Name} not found in configuration</exception>
        private OnnxModelSet LoadModelSet(IOnnxModelSetConfig modelSetConfig)
        {
            if (_onnxModelSets.ContainsKey(modelSetConfig))
                return _onnxModelSets[modelSetConfig];
          
            if (!modelSetConfig.IsEnabled)
                throw new Exception($"Model {modelSetConfig.Name} is not enabled");

            var modelSet = new OnnxModelSet(modelSetConfig);
            _onnxModelSets.TryAdd(modelSetConfig, modelSet);
            return modelSet;
        }


        /// <summary>
        /// Unloads the model set.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        private bool UnloadModelSet(IOnnxModel model)
        {
            if (!_onnxModelSets.TryGetValue(model, out _))
                return true;

            if (_onnxModelSets.TryRemove(model, out var modelSet))
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
