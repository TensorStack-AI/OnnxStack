using OnnxStack.StableDiffusion.AMD.StableDiffusion;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;

namespace OnnxStack.Console.Runner
{
    public sealed class AMDExample : IExampleRunner
    {
        private readonly string _outputDirectory;

        public AMDExample()
        {
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(AMDExample));
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 100;

        public string Name => "AMD Pipeline Example";

        public string Description => "AMD Pipeline Example";

        public async Task RunAsync()
        {
            // Create SD Pipeline
            var provider = Providers.RyzenAI(0, PipelineType.StableDiffusion);
            var pipeline = AMDNPUStableDiffusionPipeline.CreatePipeline(provider, "C:\\Models\\stable-diffusion-1.5_io32_amdgpu_npu_v0530");
            var scheduler = SchedulerType.EulerAncestral;
            //var inputImage = await OnnxImage.FromFileAsync("C:\\Models\\Depth\\Sample2.png");
            //var controlNet = ControlNetModel.Create(provider, "C:\\Models\\Depth\\model.onnx");


            // Create SD3 Pipeline
            //var provider = Providers.RyzenAI(0, PipelineType.StableDiffusion3);
            //var pipeline = AMDNPUStableDiffusion3Pipeline.CreatePipeline(provider, "C:\\Models\\stable-diffusion-3-medium_amdgpu_npu_v0530");
            //var scheduler = SchedulerType.FlowMatchEulerDiscrete;


            // Prompt
            var generateOptions = new GenerateOptions
            {
                Prompt = "An astronaut riding a horse",
                Diffuser = DiffuserType.TextToImage,

                //Diffuser = DiffuserType.ImageToImage,
                //InputImage = inputImage,

                //Diffuser = DiffuserType.ControlNet,
                //InputContolImage = inputImage,
                //ControlNet = controlNet

                // Scheduler
                SchedulerOptions = pipeline.DefaultSchedulerOptions with
                {
                    SchedulerType = scheduler
                }
            };


            // Run pipeline
            var result = await pipeline.GenerateAsync(generateOptions, progressCallback: OutputHelpers.ProgressCallback);

            // Save Image File
            await result.SaveAsync(Path.Combine(_outputDirectory, $"{generateOptions.SchedulerOptions.Seed}.png"));

            //Unload
            await pipeline.UnloadAsync();
        }
    }
}
