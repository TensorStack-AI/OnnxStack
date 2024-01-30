using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;

namespace OnnxStack.ImageUpscaler.Common
{
    public class UpscaleModel : OnnxModelSession
    {
        private readonly int _channels;
        private readonly int _sampleSize;
        private readonly int _scaleFactor;

        public UpscaleModel(UpscaleModelConfig configuration) : base(configuration)
        {
            _channels = configuration.Channels;
            _sampleSize = configuration.SampleSize;
            _scaleFactor = configuration.ScaleFactor;
        }

        public int Channels => _channels;
        public int SampleSize => _sampleSize;
        public int ScaleFactor => _scaleFactor;


        public static UpscaleModel Create(UpscaleModelConfig configuration)
        {
            return new UpscaleModel(configuration);
        }

        public static UpscaleModel Create(string modelFile, int scaleFactor, int sampleSize = 512, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML)
        {
            var configuration = new UpscaleModelConfig
            {
                Channels = 3,
                SampleSize = sampleSize,
                ScaleFactor = scaleFactor,
                DeviceId = deviceId,
                ExecutionProvider = executionProvider,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                InterOpNumThreads = 0,
                IntraOpNumThreads = 0,
                OnnxModelPath = modelFile
            };
            return new UpscaleModel(configuration);
        }
    }


    public record UpscaleModelConfig : OnnxModelConfig
    {
        public int Channels { get; set; }
        public int SampleSize { get; set; }
        public int ScaleFactor { get; set; }
    }
}
