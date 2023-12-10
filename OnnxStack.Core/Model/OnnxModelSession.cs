using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using System;
using System.IO;

namespace OnnxStack.Core.Model
{
    public class OnnxModelSession : IDisposable
    {
        private readonly SessionOptions _options;
        private readonly InferenceSession _session;
        private readonly OnnxModelConfig _configuration;

        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxModelSession"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="container">The container.</param>
        /// <exception cref="System.IO.FileNotFoundException">Onnx model file not found</exception>
        public OnnxModelSession(OnnxModelConfig configuration, PrePackedWeightsContainer container)
        {
            if (!File.Exists(configuration.OnnxModelPath))
                throw new FileNotFoundException("Onnx model file not found", configuration.OnnxModelPath);

            _configuration = configuration;
            _options = configuration.GetSessionOptions();
            _options.RegisterOrtExtensions();
            _session = new InferenceSession(_configuration.OnnxModelPath, _options, container);
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
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            _options?.Dispose();
            _session?.Dispose();
        }
    }
}
