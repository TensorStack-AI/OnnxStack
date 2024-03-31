using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;

namespace OnnxStack.FeatureExtractor.Common
{
    public class FeatureExtractorModel : OnnxModelSession
    {
        private readonly FeatureExtractorModelConfig _configuration;

        public FeatureExtractorModel(FeatureExtractorModelConfig configuration)
            : base(configuration)
        {
            _configuration = configuration;
        }

        public int OutputChannels => _configuration.OutputChannels;
        public int SampleSize => _configuration.SampleSize;
        public bool NormalizeOutput => _configuration.NormalizeOutput;
        public bool SetOutputToInputAlpha => _configuration.SetOutputToInputAlpha;
        public ImageResizeMode InputResizeMode => _configuration.InputResizeMode;
        public ImageNormalizeType NormalizeType => _configuration.NormalizeType;
        public bool NormalizeInput => _configuration.NormalizeInput;

        public static FeatureExtractorModel Create(FeatureExtractorModelConfig configuration)
        {
            return new FeatureExtractorModel(configuration);
        }

        public static FeatureExtractorModel Create(string modelFile, int sampleSize = 0, int outputChannels = 1, ImageNormalizeType normalizeType = ImageNormalizeType.ZeroToOne, bool normalizeInput = true, bool normalizeOutput = false, ImageResizeMode inputResizeMode = ImageResizeMode.Crop, bool setOutputToInputAlpha = false, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML)
        {
            var configuration = new FeatureExtractorModelConfig
            {
                DeviceId = deviceId,
                ExecutionProvider = executionProvider,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                InterOpNumThreads = 0,
                IntraOpNumThreads = 0,
                OnnxModelPath = modelFile,

                SampleSize = sampleSize,
                OutputChannels = outputChannels,
                NormalizeType = normalizeType,
                NormalizeInput = normalizeInput,
                NormalizeOutput = normalizeOutput,
                SetOutputToInputAlpha = setOutputToInputAlpha,
                InputResizeMode = inputResizeMode
            };
            return new FeatureExtractorModel(configuration);
        }
    }
}
