using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using System;

namespace OnnxStack.Core.Model
{
    public class OnnxExecutionProvider
    {
        private readonly string _name;
        private readonly Func<OnnxModelConfig, SessionOptions> _sessionOptionsFactory;

        public OnnxExecutionProvider(string name, Func<OnnxModelConfig, SessionOptions> sessionOptionsFactory)
        {
            _name = name;
            _sessionOptionsFactory = sessionOptionsFactory;
        }

        public string Name => _name;

        public SessionOptions CreateSession(OnnxModelConfig modelConfig)
        {
            return _sessionOptionsFactory(modelConfig);
        }
    }
}
