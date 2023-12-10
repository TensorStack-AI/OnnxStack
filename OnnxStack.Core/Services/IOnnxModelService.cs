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
        /// Gets the active loaded ModelSets.
        /// </summary>
        IEnumerable<OnnxModelSet> ModelSets { get; }

        /// <summary>
        /// Gets the ModelSet configs.
        /// </summary>
        IEnumerable<IOnnxModelSetConfig> ModelSetConfigs { get; }

        /// <summary>
        /// Adds a ModelSet
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <returns></returns>
        Task<bool> AddModelSet(IOnnxModelSetConfig modelSet);


        /// <summary>
        /// Adds a collection of ModelSet
        /// </summary>
        /// <param name="modelSets">The model sets.</param>
        Task AddModelSet(IEnumerable<IOnnxModelSetConfig> modelSets);


        /// <summary>
        /// Removes a model set.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <returns></returns>
        Task<bool> RemoveModelSet(IOnnxModelSetConfig modelSet);

        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        Task<OnnxModelSet> LoadModelAsync(IOnnxModel model);

        /// <summary>
        /// Unloads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        Task<bool> UnloadModelAsync(IOnnxModel model);

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
        /// Runs the inference Use when output size is unknown
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="inputs">The inputs.</param>
        /// <param name="outputs">The outputs.</param>
        /// <returns></returns>
        IDisposableReadOnlyCollection<OrtValue> RunInference(IOnnxModel model, OnnxModelType modelType, OnnxInferenceParameters parameters);


        /// <summary>
        /// Runs the inference asynchronously, Use when output size is known
        /// Output buffer size must be known and set before inference is run
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="inputs">The inputs.</param>
        /// <param name="outputs">The outputs.</param>
        /// <returns></returns>
        Task<IReadOnlyCollection<OrtValue>> RunInferenceAsync(IOnnxModel model, OnnxModelType modelType, OnnxInferenceParameters parameters);


        /// <summary>
        /// Gets the model metadata.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <returns></returns>
        OnnxMetadata GetModelMetadata(IOnnxModel model, OnnxModelType modelType);
    }
}