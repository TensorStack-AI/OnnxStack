using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Services;

namespace OnnxStack.Console.Runner
{
    public sealed class LoRADebug : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly IOnnxModelService _modelService;
        private readonly IOnnxModelAdaptaterService _modelAdaptaterService;

        public LoRADebug(IOnnxModelService modelService)
        {
            _modelService = modelService;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(StableDebug));
        }

        public string Name => "LoRA Debug";

        public string Description => "LoRA Debugger";

        public async Task RunAsync()
        {
            string modelPath = "D:\\Repositories\\stable-diffusion-v1-5\\unet\\model.onnx";
            string loraModelPath = "D:\\Repositories\\LoRAFiles\\model.onnx";

            using (var modelession = new InferenceSession(modelPath))
            using (var loraModelSession = new InferenceSession(loraModelPath))
            {
                try
                {
                    _modelAdaptaterService.ApplyLowRankAdaptation(modelession, loraModelSession);
                }
                catch (Exception ex)
                {

                }
            }
        }

    }
}
