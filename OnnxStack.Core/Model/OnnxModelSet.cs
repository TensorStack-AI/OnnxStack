using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using System;
using System.Collections.Immutable;
using System.Linq;

namespace OnnxStack.Core.Model
{
    public class OnnxModelSet : IDisposable
    {
        private readonly IOnnxModelSetConfig _configuration;
        private readonly PrePackedWeightsContainer _prePackedWeightsContainer;
        private readonly ImmutableDictionary<OnnxModelType, OnnxModelSession> _modelSessions;

        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxModelSet"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public OnnxModelSet(IOnnxModelSetConfig configuration)
        {
            _configuration = configuration;
            _prePackedWeightsContainer = new PrePackedWeightsContainer();
            _modelSessions = configuration.ModelConfigurations.ToImmutableDictionary(
                modelConfig => modelConfig.Type,
                modelConfig => new OnnxModelSession(modelConfig, _prePackedWeightsContainer));
        }


        /// <summary>
        /// Gets the name.
        /// </summary>
        public string Name => _configuration.Name;
      

        /// <summary>
        /// Checks the specified model type exists in the set.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <returns></returns>
        public bool Exists(OnnxModelType modelType)
        {
            return _modelSessions.ContainsKey(modelType);
        }


        /// <summary>
        /// Gets the options.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <returns></returns>
        /// <exception cref="System.Exception">Model {modelType} not found</exception>
        public SessionOptions GetOptions(OnnxModelType modelType)
        {
            if (!Exists(modelType))
                throw new Exception($"Model {modelType} not found");

            return _modelSessions[modelType].Options;
        }


        /// <summary>
        /// Gets the InferenceSession.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <returns></returns>
        /// <exception cref="System.Exception">Model {modelType} not found</exception>
        public InferenceSession GetSession(OnnxModelType modelType)
        {
            if (!Exists(modelType))
                throw new Exception($"Model {modelType} not found");

            return _modelSessions[modelType].Session;
        }


        /// <summary>
        /// Gets the configuration.
        /// </summary>
        /// <param name="modelType">Type of the model.</param>
        /// <returns></returns>
        public OnnxModelConfig GetConfiguration(OnnxModelType modelType)
        {
            return _configuration.ModelConfigurations.FirstOrDefault(x => x.Type == modelType);
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            foreach (var modelSession in _modelSessions.Values)
            {
                modelSession?.Dispose();
            }
            _prePackedWeightsContainer?.Dispose();
        }
    }
}
