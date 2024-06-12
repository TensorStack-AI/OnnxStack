using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;

namespace OnnxStack.StableDiffusion.Models
{
    public class AutoEncoderModel : OnnxModelSession
    {
        private readonly AutoEncoderModelConfig _configuration;

        public AutoEncoderModel(AutoEncoderModelConfig configuration) : base(configuration)
        {
            _configuration = configuration;
        }

        public float ScaleFactor => _configuration.ScaleFactor;


        public static AutoEncoderModel Create(AutoEncoderModelConfig configuration)
        {
            return new AutoEncoderModel(configuration);
        }

        public static AutoEncoderModel Create(string modelFile, float scaleFactor, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML)
        {
            var configuration = new AutoEncoderModelConfig
            {
                DeviceId = deviceId,
                ExecutionProvider = executionProvider,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                InterOpNumThreads = 0,
                IntraOpNumThreads = 0,
                OnnxModelPath = modelFile,
                ScaleFactor = scaleFactor
            };
            return new AutoEncoderModel(configuration);
        }
    }

    public record AutoEncoderModelConfig : OnnxModelConfig
    {
        public float ScaleFactor { get; set; }
    }
}
