using OnnxStack.Core.Config;
using OnnxStack.Core.Model;

namespace OnnxStack.ImageUpscaler.Models
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
        public int ScaleFactor  => _scaleFactor;
    }


    public record UpscaleModelConfig : OnnxModelConfig
    {
        public int Channels { get; set; }
        public int SampleSize { get; set; }
        public int ScaleFactor { get; set; }
    }
}
