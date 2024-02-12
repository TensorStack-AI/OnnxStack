using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;

namespace OnnxStack.FeatureExtractor.Common
{
    public class FeatureExtractorModel : OnnxModelSession
    {
        private readonly int _sampleSize;
        private readonly bool _normalize;
        private readonly int _channels;

        public FeatureExtractorModel(FeatureExtractorModelConfig configuration)
            : base(configuration)
        {
            _sampleSize = configuration.SampleSize;
            _normalize = configuration.Normalize;
            _channels = configuration.Channels;
        }

        public int SampleSize => _sampleSize;

        public bool Normalize => _normalize;

        public int Channels => _channels;

        public static FeatureExtractorModel Create(FeatureExtractorModelConfig configuration)
        {
            return new FeatureExtractorModel(configuration);
        }

        public static FeatureExtractorModel Create(string modelFile, bool normalize = false, int sampleSize = 512, int channels = 3, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML)
        {
            var configuration = new FeatureExtractorModelConfig
            {
                SampleSize = sampleSize,
                Normalize = normalize,
                Channels = channels,
                DeviceId = deviceId,
                ExecutionProvider = executionProvider,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                InterOpNumThreads = 0,
                IntraOpNumThreads = 0,
                OnnxModelPath = modelFile
            };
            return new FeatureExtractorModel(configuration);
        }
    }

    public record FeatureExtractorModelConfig : OnnxModelConfig
    {
        public int SampleSize { get; set; }
        public bool Normalize { get; set; }
        public int Channels { get; set; }
    }
}
