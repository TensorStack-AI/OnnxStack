using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using SixLabors.ImageSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.Core.Model
{
    public class OnnxModelSession : IDisposable
    {
        private readonly OnnxModelConfig _configuration;

        private SessionOptions _options;
        private OnnxMetadata _metadata;
        private InferenceSession _session;
        private OnnxOptimizations _optimizations;

        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxModelSession"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <exception cref="System.IO.FileNotFoundException">Onnx model file not found</exception>
        public OnnxModelSession(OnnxModelConfig configuration)
        {
            ArgumentNullException.ThrowIfNull(configuration.ExecutionProvider);
            if (!File.Exists(configuration.OnnxModelPath))
                throw new FileNotFoundException($"Onnx model file not found, Path: {configuration.OnnxModelPath}", configuration.OnnxModelPath);

            _configuration = configuration;
        }


        /// <summary>
        /// Gets the SessionOptions.
        /// </summary>
        public SessionOptions Options => _options;


        /// <summary>
        /// Gets the InferenceSession.
        /// </summary>
        public InferenceSession Session => _session;


        /// <summary>
        /// Gets the configuration.
        /// </summary>
        public OnnxModelConfig Configuration => _configuration;


        /// <summary>
        /// Loads the model session.
        /// </summary>
        public async Task<OnnxMetadata> LoadAsync(OnnxOptimizations optimizations = default, CancellationToken cancellationToken = default)
        {
            try
            {
                if (_session is null)
                    return await CreateSession(optimizations, cancellationToken);

                if (HasOptimizationsChanged(optimizations))
                {
                    await UnloadAsync();
                    return await CreateSession(optimizations, cancellationToken);
                }
                return _metadata;
            }
            catch (OnnxRuntimeException ex)
            {
                if (ex.Message.Contains("ErrorCode:RequirementNotRegistered"))
                    throw new OperationCanceledException("Inference was canceled.", ex);
                throw;
            }
        }


        /// <summary>
        /// Unloads the model session.
        /// </summary>
        /// <returns></returns>
        public async Task UnloadAsync()
        {
            // TODO: deadlock on model dispose when no synchronization context exists(console app)
            // Task.Yield seems to force a context switch resolving any issues, revist this
            await Task.Yield();

            if (_session is not null)
            {
                _session.Dispose();
                _metadata = null;
                _session = null;
            }
        }


        /// <summary>
        /// Runs inference on the model with the suppied parameters, use this method when you do not have a known output shape.
        /// </summary>
        /// <param name="parameters">The parameters.</param>
        /// <returns></returns>
        public IDisposableReadOnlyCollection<OrtValue> RunInference(OnnxInferenceParameters parameters)
        {
            try
            {
                return _session.Run(parameters.RunOptions, parameters.InputNameValues, parameters.OutputNames);
            }
            catch (OnnxRuntimeException ex)
            {
                if (ex.Message.Contains("Exiting due to terminate flag"))
                    throw new OperationCanceledException("Inference was canceled.", ex);
                throw;
            }
        }


        /// <summary>
        /// Runs inference on the model with the suppied parameters, use this method when the output shape is known
        /// </summary>
        /// <param name="parameters">The parameters.</param>
        /// <returns></returns>
        public async Task<IReadOnlyCollection<OrtValue>> RunInferenceAsync(OnnxInferenceParameters parameters)
        {
            try
            {
                return await _session.RunAsync(parameters.RunOptions, parameters.InputNames, parameters.InputValues, parameters.OutputNames, parameters.OutputValues);
            }
            catch (OnnxRuntimeException ex)
            {
                if (ex.Message.Contains("Exiting due to terminate flag"))
                    throw new OperationCanceledException("Inference was canceled.", ex);
                throw;
            }
        }


        /// <summary>
        /// Creates the InferenceSession.
        /// </summary>
        /// <param name="optimizations">The optimizations.</param>
        /// <returns>The Sessions OnnxMetadata.</returns>
        private async Task<OnnxMetadata> CreateSession(OnnxOptimizations optimizations, CancellationToken cancellationToken)
        {
            _options?.Dispose();
            _options = _configuration.ExecutionProvider.CreateSession(_configuration);
            cancellationToken.Register(_options.CancelSession, true);

            if (_configuration.IsOptimizationSupported)
                ApplyOptimizations(optimizations);

            _session = await Task.Run(() => new InferenceSession(_configuration.OnnxModelPath, _options), cancellationToken);
            _metadata = new OnnxMetadata
            {
                Inputs = _session.InputMetadata.Select(OnnxNamedMetadata.Create).ToList(),
                Outputs = _session.OutputMetadata.Select(OnnxNamedMetadata.Create).ToList()
            };
            return _metadata;
        }


        /// <summary>
        /// Applies the optimizations.
        /// </summary>
        /// <param name="optimizations">The optimizations.</param>
        private void ApplyOptimizations(OnnxOptimizations optimizations)
        {
            _optimizations = optimizations;
            if (_optimizations != null)
            {
                _options.GraphOptimizationLevel = optimizations.GraphOptimizationLevel;
                if (_optimizations.GraphOptimizationLevel == GraphOptimizationLevel.ORT_DISABLE_ALL)
                    return;

                foreach (var freeDimensionOverride in _optimizations.DimensionOverrides)
                {
                    if (freeDimensionOverride.Key.StartsWith("dummy_"))
                        continue;

                    _options.AddFreeDimensionOverrideByName(freeDimensionOverride.Key, freeDimensionOverride.Value);
                }
            }
        }


        /// <summary>
        /// Determines whether optimizations have changed
        /// </summary>
        /// <param name="optimizations">The optimizations.</param>
        /// <returns><c>true</c> if changed; otherwise, <c>false</c>.</returns>
        public bool HasOptimizationsChanged(OnnxOptimizations optimizations)
        {
            if (_optimizations == null && optimizations == null)
                return false; // No Optimizations set

            if (_optimizations == optimizations)
                return false; // Optimizations have not changed

            return true;
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            _options?.Dispose();
            _session?.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}
