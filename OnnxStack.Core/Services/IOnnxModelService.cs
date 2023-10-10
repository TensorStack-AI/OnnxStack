using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace OnnxStack.Core.Services
{
    public interface IOnnxModelService : IDisposable
    {

        /// <summary>
        /// Gets the configuration.
        /// </summary>
        OnnxStackConfig Configuration { get; }

        /// <summary>
        /// Determines whether the specified model type is enabled.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <returns>
        ///   <c>true</c> if the specified model type is enabled; otherwise, <c>false</c>.
        /// </returns>
        bool IsEnabled(OnnxModelType modelType);

        /// <summary>
        /// Determines whether the specified model type is enabled.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <returns>
        ///   <c>true</c> if the specified model type is enabled; otherwise, <c>false</c>.
        /// </returns>
        Task<bool> IsEnabledAsync(OnnxModelType modelType);


        /// <summary>
        /// Runs inference on the specified model.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="inputs">The inputs.</param>
        /// <returns></returns>
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunInference(OnnxModelType modelType, IReadOnlyCollection<NamedOnnxValue> inputs);


        /// <summary>
        /// Runs inference on the specified model.asynchronously.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="inputs">The inputs.</param>
        /// <returns></returns>
        Task<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> RunInferenceAsync(OnnxModelType modelType, IReadOnlyCollection<NamedOnnxValue> inputs);
    }
}