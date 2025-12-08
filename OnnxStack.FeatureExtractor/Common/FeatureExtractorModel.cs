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
        public bool SetOutputToInputAlpha => _configuration.SetOutputToInputAlpha;
        public ImageResizeMode InputResizeMode => _configuration.InputResizeMode;
        public ImageNormalizeType NormalizeType => _configuration.NormalizeType;
        public ImageNormalizeType NormalizeOutputType => _configuration.NormalizeOutputType;
        public bool InvertOutput => _configuration.InvertOutput;


        public static FeatureExtractorModel Create(FeatureExtractorModelConfig configuration)
        {
            return new FeatureExtractorModel(configuration);
        }


        public static FeatureExtractorModel Create(OnnxExecutionProvider executionProvider, string modelFile, int sampleSize = 0, int outputChannels = 1, ImageNormalizeType normalizeType = ImageNormalizeType.None, ImageNormalizeType normalizeOutputType = ImageNormalizeType.None, ImageResizeMode inputResizeMode = ImageResizeMode.Crop, bool setOutputToInputAlpha = false, bool invertOutput = false)
        {
            var configuration = new FeatureExtractorModelConfig
            {
                OnnxModelPath = modelFile,
                ExecutionProvider = executionProvider,
                SampleSize = sampleSize,
                OutputChannels = outputChannels,
                NormalizeType = normalizeType,
                NormalizeOutputType = normalizeOutputType,
                SetOutputToInputAlpha = setOutputToInputAlpha,
                InputResizeMode = inputResizeMode,
                InvertOutput = invertOutput
            };
            return new FeatureExtractorModel(configuration);
        }
    }
}
