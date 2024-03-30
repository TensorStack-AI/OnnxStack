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
        public bool NormalizeOutputTensor => _configuration.NormalizeOutputTensor;
        public bool SetOutputToInputAlpha => _configuration.SetOutputToInputAlpha;
        public ImageResizeMode InputResizeMode => _configuration.InputResizeMode;
        public ImageNormalizeType InputNormalization => _configuration.NormalizeInputTensor;

        public static FeatureExtractorModel Create(FeatureExtractorModelConfig configuration)
        {
            return new FeatureExtractorModel(configuration);
        }

        public static FeatureExtractorModel Create(string modelFile, int sampleSize = 0, int outputChannels = 1, bool normalizeOutputTensor = false, ImageNormalizeType normalizeInputTensor = ImageNormalizeType.ZeroToOne, ImageResizeMode inputResizeMode = ImageResizeMode.Crop, bool setOutputToInputAlpha = false, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML)
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
                NormalizeOutputTensor = normalizeOutputTensor,
                SetOutputToInputAlpha = setOutputToInputAlpha,
                NormalizeInputTensor = normalizeInputTensor,
                InputResizeMode = inputResizeMode
            };
            return new FeatureExtractorModel(configuration);
        }
    }
}
