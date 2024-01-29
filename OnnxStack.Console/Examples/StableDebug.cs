using OnnxStack.Core.Image;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Pipelines;
using SixLabors.ImageSharp;
using System.Diagnostics;

namespace OnnxStack.Console.Runner
{
    public sealed class StableDebug : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly StableDiffusionConfig _configuration;

        public StableDebug(StableDiffusionConfig configuration)
        {
            _configuration = configuration;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(StableDebug));
        }

        public int Index => 0;

        public string Name => "Stable Diffusion Debug";

        public string Description => "Stable Diffusion Debugger";

        /// <summary>
        /// Stable Diffusion Demo
        /// </summary>
        public async Task RunAsync()
        {
            Directory.CreateDirectory(_outputDirectory);

            var prompt = "High-fashion photography in an abandoned industrial warehouse, with dramatic lighting and edgy outfits, detailed clothing, intricate clothing, seductive pose, action pose, motion, beautiful digital artwork, atmospheric, warm sunlight, photography, neo noir, bokeh, beautiful dramatic lighting, shallow depth of field, photorealism, volumetric lighting, Ultra HD, raytracing, studio quality, octane render";
            var negativePrompt = "painting, drawing, sketches, monochrome, grayscale, illustration, anime, cartoon, graphic, text, crayon, graphite, abstract, easynegative, low quality, normal quality, worst quality, lowres, close up, cropped, out of frame, jpeg artifacts, duplicate, morbid, mutilated, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, glitch, deformed, mutated, cross-eyed, ugly, dehydrated, bad anatomy, bad proportions, gross proportions, cloned face, disfigured, malformed limbs, missing arms, missing legs fused fingers, too many fingers,extra fingers, extra limbs,, extra arms, extra legs,disfigured,";
            while (true)
            {
                var developmentSeed = 624461087;
                var promptOptions = new PromptOptions
                {
                    Prompt = prompt,
                    NegativePrompt = negativePrompt,
                };

                // Loop though the appsettings.json model sets
                foreach (var modelSet in _configuration.ModelSets)
                {
                    OutputHelpers.WriteConsole($"Loading Model `{modelSet.Name}`...", ConsoleColor.Cyan);

                    // Create Pipeline
                    var pipeline = PipelineBase.CreatePipeline(modelSet);

                    // Preload Models (optional)
                    await pipeline.LoadAsync();

                    // Loop though schedulers
                    foreach (var scheduler in pipeline.SupportedSchedulers)
                    {
                        // Create SchedulerOptions based on pipeline defaults
                        var schedulerOptions = pipeline.DefaultSchedulerOptions with
                        {
                            Seed = developmentSeed,
                            SchedulerType = scheduler
                        };

                        var timestamp = Stopwatch.GetTimestamp();
                        OutputHelpers.WriteConsole($"Generating {scheduler} Image...", ConsoleColor.Green);

                        // Run pipeline
                        var result = await pipeline.RunAsync(promptOptions, schedulerOptions, progressCallback: OutputHelpers.ProgressCallback);

                        // Create Image from Tensor result
                        var image = result.ToImage();

                        // Save Image File
                        var outputFilename = Path.Combine(_outputDirectory, $"{modelSet.Name}_{schedulerOptions.SchedulerType}.png");
                        await image.SaveAsPngAsync(outputFilename);

                        OutputHelpers.WriteConsole($"{schedulerOptions.SchedulerType} Image Created: {Path.GetFileName(outputFilename)}", ConsoleColor.Green);
                        OutputHelpers.WriteConsole($"Elapsed: {Stopwatch.GetElapsedTime(timestamp)}ms", ConsoleColor.Yellow);
                    }

                    OutputHelpers.WriteConsole($"Unloading Model `{modelSet.Name}`...", ConsoleColor.Cyan);

                    // Unload pipeline
                    await pipeline.UnloadAsync();
                }
                break;
            }
        }
    }
}
