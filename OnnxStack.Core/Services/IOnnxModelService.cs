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