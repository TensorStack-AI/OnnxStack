using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace OnnxStack.Core.Services
{
    public interface IOnnxModelService : IDisposable
    {

        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        Task<OnnxModelSet> LoadModel(IOnnxModel model);

        /// <summary>
        /// Unloads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        Task<bool> UnloadModel(IOnnxModel model);

        /// <summary>
        /// Determines whether the specified model is loaded.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns>
        ///   <c>true</c> if the specified model is loaded; otherwise, <c>false</c>.
        /// </returns>
        bool IsModelLoaded(IOnnxModel model);


        /// <summary>
        /// Updates the model set.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <returns></returns>
        bool UpdateModelSet(IOnnxModelSetConfig modelSet);

        /// <summary>
        /// Determines whether the specified model type is enabled.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <returns>
        ///   <c>true</c> if the specified model type is enabled; otherwise, <c>false</c>.
        /// </returns>
        bool IsEnabled(IOnnxModel model, OnnxModelType modelType);

        /// <summary>
        /// Determines whether the specified model type is enabled.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <returns>
        ///   <c>true</c> if the specified model type is enabled; otherwise, <c>false</c>.
        /// </returns>
        Task<bool> IsEnabledAsync(IOnnxModel model, OnnxModelType modelType);


        /// <summary>
        /// Runs inference on the specified model.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="inputs">The inputs.</param>
        /// <returns></returns>
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunInference(IOnnxModel model, OnnxModelType modelType, IReadOnlyCollection<NamedOnnxValue> inputs);


        /// <summary>
        /// Runs inference on the specified model.asynchronously.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="inputs">The inputs.</param>
        /// <returns></returns>
        Task<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> RunInferenceAsync(IOnnxModel model, OnnxModelType modelType, IReadOnlyCollection<NamedOnnxValue> inputs);


        /// <summary>
        /// Runs the inference Use when output size is unknown
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="inputs">The inputs.</param>
        /// <param name="outputs">The outputs.</param>
        /// <returns></returns>
        IReadOnlyCollection<OrtValue> RunInference(IOnnxModel model, OnnxModelType modelType, Dictionary<string, OrtValue> inputs, IReadOnlyCollection<string> outputs);


        /// <summary>
        /// Runs the inference asynchronously, Use when output size is known
        /// Output buffer size must be known and set before inference is run
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="inputs">The inputs.</param>
        /// <param name="outputs">The outputs.</param>
        /// <returns></returns>
        Task<IReadOnlyCollection<OrtValue>> RunInferenceAsync(IOnnxModel model, OnnxModelType modelType, Dictionary<string, OrtValue> inputs, Dictionary<string, OrtValue> outputs);


        /// <summary>
        /// Gets the Sessions input metadata.
        /// </summary>
        /// <param name="modelType">Type of model.</param>
        /// <returns></returns>
        IReadOnlyDictionary<string, NodeMetadata> GetInputMetadata(IOnnxModel model, OnnxModelType modelType);


        /// <summary>
        /// Gets the Sessions input names.
        /// </summary>
        /// <param name="modelType">Type of model.</param>
        /// <returns></returns>
        IReadOnlyList<string> GetInputNames(IOnnxModel model, OnnxModelType modelType);


        /// <summary>
        /// Gets the Sessions output metadata.
        /// </summary>
        /// <param name="modelType">Type of model.</param>
        /// <returns></returns>
        IReadOnlyDictionary<string, NodeMetadata> GetOutputMetadata(IOnnxModel model, OnnxModelType modelType);


        /// <summary>
        /// Gets the Sessions output metadata names.
        /// </summary>
        /// <param name="modelType">Type of model.</param>
        /// <returns></returns>
        IReadOnlyList<string> GetOutputNames(IOnnxModel model, OnnxModelType modelType);

    }
}