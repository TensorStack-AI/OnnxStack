using OnnxStack.Core.Image;
using OnnxStack.Core.Model;

namespace OnnxStack.ImageUpscaler.Common
{
    public class UpscaleModel : OnnxModelSession
    {
        private readonly UpscaleModelConfig _configuration;

        public UpscaleModel(UpscaleModelConfig configuration)
            : base(configuration)
        {
            _configuration = configuration;
        }

        public int Channels => _configuration.Channels;
        public int SampleSize => _configuration.SampleSize;
        public int ScaleFactor => _configuration.ScaleFactor;
        public ImageNormalizeType NormalizeType => _configuration.NormalizeType;


        public static UpscaleModel Create(UpscaleModelConfig configuration)
        {
            return new UpscaleModel(configuration);
        }


        public static UpscaleModel Create(OnnxExecutionProvider executionProvider, string modelFile, int scaleFactor, int sampleSize, ImageNormalizeType normalizeType = ImageNormalizeType.ZeroToOne, int channels = 3)
        {
            var configuration = new UpscaleModelConfig
            {
                Channels = channels,
                SampleSize = sampleSize,
                ScaleFactor = scaleFactor,
                NormalizeType = normalizeType,
                OnnxModelPath = modelFile,
                ExecutionProvider = executionProvider
            };
            return new UpscaleModel(configuration);
        }
    }
}
