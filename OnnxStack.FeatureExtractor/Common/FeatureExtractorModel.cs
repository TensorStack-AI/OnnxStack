using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;

namespace OnnxStack.FeatureExtractor.Common
{
    public class FeatureExtractorModel : OnnxModelSession
    {
        private readonly int _sampleSize;

        public FeatureExtractorModel(FeatureExtractorModelConfig configuration)
            : base(configuration)
        {
            _sampleSize = configuration.SampleSize;
        }

        public int SampleSize => _sampleSize;

        public static FeatureExtractorModel Create(FeatureExtractorModelConfig configuration)
        {
            return new FeatureExtractorModel(configuration);
        }

        public static FeatureExtractorModel Create(string modelFile, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML)
        {
            var configuration = new FeatureExtractorModelConfig
            {
                SampleSize = 512,
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
    }
}
