using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace OnnxStack.Core.Model
{
    public class OnnxModelSession : IDisposable
    {
        private readonly SessionOptions _options;
        private readonly OnnxModelConfig _configuration;

        private OnnxMetadata _metadata;
        private InferenceSession _session;


        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxModelSession"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <exception cref="System.IO.FileNotFoundException">Onnx model file not found</exception>
        public OnnxModelSession(OnnxModelConfig configuration)
        {
            if (!File.Exists(configuration.OnnxModelPath))
                throw new FileNotFoundException("Onnx model file not found", configuration.OnnxModelPath);

            _configuration = configuration;
            _options = configuration.GetSessionOptions();
            _options.RegisterOrtExtensions();
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
        public async Task LoadAsync()
        {
            if (_session is not null)
                return; // Already Loaded

            _session = await Task.Run(() => new InferenceSession(_configuration.OnnxModelPath, _options));
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
        /// Gets the metadata.
        /// </summary>
        /// <returns></returns>
        public async Task<OnnxMetadata> GetMetadataAsync()
        {
            if (_metadata is not null)
                return _metadata;

            if (_session is null)
                await LoadAsync();

            _metadata = new OnnxMetadata
            {
                Inputs = _session.InputMetadata.Select(OnnxNamedMetadata.Create).ToList(),
                Outputs = _session.OutputMetadata.Select(OnnxNamedMetadata.Create).ToList()
            };
            return _metadata;
        }


        /// <summary>
        /// Runs inference on the model with the suppied parameters, use this method when you do not have a known output shape.
        /// </summary>
        /// <param name="parameters">The parameters.</param>
        /// <returns></returns>
        public IDisposableReadOnlyCollection<OrtValue> RunInference(OnnxInferenceParameters parameters)
        {
            return _session.Run(parameters.RunOptions, parameters.InputNameValues, parameters.OutputNames);
        }


        /// <summary>
        /// Runs inference on the model with the suppied parameters, use this method when the output shape is known
        /// </summary>
        /// <param name="parameters">The parameters.</param>
        /// <returns></returns>
        public Task<IReadOnlyCollection<OrtValue>> RunInferenceAsync(OnnxInferenceParameters parameters)
        {
            return _session.RunAsync(parameters.RunOptions, parameters.InputNames, parameters.InputValues, parameters.OutputNames, parameters.OutputValues);
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            _options?.Dispose();
            _session?.Dispose();
        }
    }
}
