using OnnxStack.Core.Image;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Pipelines;
using System.Diagnostics;

namespace OnnxStack.Console.Runner
{
    public sealed class StableDebug : IExampleRunner
    {
        private readonly string _outputDirectory;

        public StableDebug()
        {
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(StableDebug));
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 0;

        public string Name => "Stable Diffusion Debug";

        public string Description => "Stable Diffusion Debugger";

        /// <summary>
        /// Stable Diffusion Demo
        /// </summary>
        public async Task RunAsync()
        {
            // Execution provider
            var provider = Providers.DirectML(0);

            var developmentSeed = 624461087;
            var prompt = "High-fashion photography in an abandoned industrial warehouse, with dramatic lighting and edgy outfits, detailed clothing, intricate clothing, seductive pose, action pose, motion, beautiful digital artwork, atmospheric, warm sunlight, photography, neo noir, bokeh, beautiful dramatic lighting, shallow depth of field, photorealism, volumetric lighting, Ultra HD, raytracing, studio quality, octane render";
            var negativePrompt = "painting, drawing, sketches, monochrome, grayscale, illustration, anime, cartoon, graphic, text, crayon, graphite, abstract, easynegative, low quality, normal quality, worst quality, lowres, close up, cropped, out of frame, jpeg artifacts, duplicate, morbid, mutilated, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, glitch, deformed, mutated, cross-eyed, ugly, dehydrated, bad anatomy, bad proportions, gross proportions, cloned face, disfigured, malformed limbs, missing arms, missing legs fused fingers, too many fingers,extra fingers, extra limbs,, extra arms, extra legs,disfigured,";

            var pipelines = new IPipeline[]
            {
                LatentConsistencyPipeline.CreatePipeline(provider, "M:\\Models\\Debug\\Dreamshaper-LCM-amuse"),
                StableDiffusionPipeline.CreatePipeline(provider, "M:\\Models\\Debug\\StableDiffusion-amuse"),
                StableDiffusion2Pipeline.CreatePipeline(provider, "M:\\Models\\Debug\\StableDiffusion2-amuse"),
                StableDiffusionXLPipeline.CreatePipeline(provider, "M:\\Models\\Debug\\StableDiffusion-XL-amuse"),
                StableDiffusion3Pipeline.CreatePipeline(provider, "M:\\Models\\Debug\\StableDiffusion3-Medium-amuse"),
                StableCascadePipeline.CreatePipeline(provider, "M:\\Models\\Debug\\StableCascade-amuse"),
                FluxPipeline.CreatePipeline(provider, "M:\\Models\\Debug\\FLUX.1-Schnell-amuse")
            };

            var totalElapsed = Stopwatch.GetTimestamp();
            foreach (var pipeline in pipelines)
            {
                OutputHelpers.WriteConsole($"Loading Pipeline `{pipeline.PipelineType} - {pipeline.Name}`...", ConsoleColor.Cyan);

                var useLowMemory = pipeline.PipelineType == PipelineType.StableCascade || pipeline.PipelineType == PipelineType.Flux;

                // Create GenerateOptions
                var generateOptions = new GenerateOptions
                {
                    Prompt = prompt,
                    NegativePrompt = negativePrompt,
                    OptimizationType = OptimizationType.None,
                    IsLowMemoryComputeEnabled = useLowMemory,
                    IsLowMemoryDecoderEnabled = useLowMemory,
                    IsLowMemoryEncoderEnabled = useLowMemory,
                    IsLowMemoryTextEncoderEnabled = useLowMemory
                };

                // Loop though schedulers
                foreach (var scheduler in pipeline.SupportedSchedulers)
                {
                    // Create SchedulerOptions based on pipeline defaults
                    generateOptions.SchedulerOptions = pipeline.DefaultSchedulerOptions with
                    {
                        Seed = developmentSeed,
                        SchedulerType = scheduler
                    };

                    var timestamp = Stopwatch.GetTimestamp();
                    OutputHelpers.WriteConsole($"Generating {scheduler} Image...", ConsoleColor.Green);

                    // Run pipeline
                    var result = await pipeline.RunAsync(generateOptions, progressCallback: OutputHelpers.ProgressCallback);

                    // Create Image from Tensor result
                    var image = new OnnxImage(result);

                    // Save Image File
                    var outputFilename = Path.Combine(_outputDirectory, $"{pipeline.Name}_{generateOptions.SchedulerOptions.SchedulerType}.png");
                    await image.SaveAsync(outputFilename);

                    OutputHelpers.WriteConsole($"{generateOptions.SchedulerOptions.SchedulerType} Image Created: {Path.GetFileName(outputFilename)}", ConsoleColor.Green);
                    OutputHelpers.WriteConsole($"Elapsed: {Stopwatch.GetElapsedTime(timestamp)}ms", ConsoleColor.Yellow);

                }
                OutputHelpers.WriteConsole($"Unloading Model `{pipeline.Name}`...", ConsoleColor.Cyan);

                // Unload pipeline
                await pipeline.UnloadAsync();
            }

            OutputHelpers.WriteConsole($"Done - Elapsed: {Stopwatch.GetElapsedTime(totalElapsed)}", ConsoleColor.Yellow);
        }
    }
}
