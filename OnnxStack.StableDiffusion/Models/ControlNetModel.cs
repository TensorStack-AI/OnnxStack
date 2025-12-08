using OnnxStack.Core.Config;
using OnnxStack.Core.Model;

namespace OnnxStack.StableDiffusion.Models
{
    public class ControlNetModel : OnnxModelSession
    {
        private readonly ControlNetModelConfig _configuration;
        public ControlNetModel(ControlNetModelConfig configuration)
            : base(configuration)
        {
            _configuration = configuration;
        }

        public bool InvertInput => _configuration.InvertInput;
        public int LayerCount => _configuration.LayerCount;
        public bool DisablePooledProjection => _configuration.DisablePooledProjection;

        public static ControlNetModel Create(ControlNetModelConfig configuration)
        {
            return new ControlNetModel(configuration);
        }

        public static ControlNetModel Create(OnnxExecutionProvider executionProvider, string modelFile, bool invertInput = false, int layerCount = 0, bool disablePooledProjection = false)
        {
            var configuration = new ControlNetModelConfig
            {
                OnnxModelPath = modelFile,
                ExecutionProvider = executionProvider,
                InvertInput = invertInput,
                LayerCount = layerCount,
                DisablePooledProjection = disablePooledProjection
            };
            return new ControlNetModel(configuration);
        }
    }

    public record ControlNetModelConfig : OnnxModelConfig
    {
        public string Name { get; set; }
        public bool InvertInput { get; set; }
        public int LayerCount { get; set; }
        public bool DisablePooledProjection { get; set; }
    }
}
