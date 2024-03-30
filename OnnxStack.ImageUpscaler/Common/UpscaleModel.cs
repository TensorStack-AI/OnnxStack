using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using System;

namespace OnnxStack.ImageUpscaler.Common
{
    public class UpscaleModel : OnnxModelSession
    {
        private readonly UpscaleModelConfig _configuration;

        public UpscaleModel(UpscaleModelConfig configuration) : base(configuration)
        {
            _configuration = configuration;
        }

        public int Channels => _configuration.Channels;
        public int SampleSize => _configuration.SampleSize;
        public int ScaleFactor => _configuration.ScaleFactor;
        public int TileSize => _configuration.TileSize;
        public int TileOverlap => _configuration.TileOverlap;

        public static UpscaleModel Create(UpscaleModelConfig configuration)
        {
            return new UpscaleModel(configuration);
        }

        public static UpscaleModel Create(string modelFile, int scaleFactor, int sampleSize, int tileSize = 0, int tileOverlap = 20, int channels = 3, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML)
        {
            var configuration = new UpscaleModelConfig
            {
                Channels = channels,
                SampleSize = sampleSize,
                ScaleFactor = scaleFactor,
                TileOverlap = tileOverlap,
                TileSize = Math.Min(sampleSize, tileSize > 0 ? tileSize : sampleSize),
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
}
