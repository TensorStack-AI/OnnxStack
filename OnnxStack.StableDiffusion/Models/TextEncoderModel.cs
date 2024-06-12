using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;

namespace OnnxStack.StableDiffusion.Models
{
    public class TextEncoderModel : OnnxModelSession
    {
        private readonly TextEncoderModelConfig _configuration;

        public TextEncoderModel(TextEncoderModelConfig configuration) : base(configuration)
        {
            _configuration = configuration;
        }

        public static TextEncoderModel Create(TextEncoderModelConfig configuration)
        {
            return new TextEncoderModel(configuration);
        }

        public static TextEncoderModel Create(string modelFile, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML)
        {
            var configuration = new TextEncoderModelConfig
            {
                DeviceId = deviceId,
                ExecutionProvider = executionProvider,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                InterOpNumThreads = 0,
                IntraOpNumThreads = 0,
                OnnxModelPath = modelFile
            };
            return new TextEncoderModel(configuration);
        }
    }

    public record TextEncoderModelConfig : OnnxModelConfig
    {

    }
}
