using OnnxStack.Core;
using OnnxStack.Core.Video;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Pipelines;
using System.Collections.ObjectModel;
using System.Diagnostics;

namespace OnnxStack.Console.Runner
{
    public sealed class LocomotionExample : IExampleRunner
    {
        private readonly string _outputDirectory;

        public LocomotionExample()
        {
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(LocomotionExample));
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 40;

        public string Name => "Locomotion Debug";

        public string Description => "Locomotion Debugger";

        /// <summary>
        /// Stable Diffusion Demo
        /// </summary>
        public async Task RunAsync()
        {
            // Execution provider
            var provider = Providers.DirectML(0);

            var totalElapsed = Stopwatch.GetTimestamp();

            var developmentSeed = Random.Shared.Next();
            var guidanceScale = 0;
            var motionFrames = 16;
            var postfix = "";
            var pipeline = LocomotionPipeline.CreatePipeline(provider, "M:\\Amuse3\\_uploaded\\Locomotion-ToonYou-amuse", 16);

            var prompts = new string[]
            {
                "Womans portrait | Womans portrait, smiling",
                "Mans portrait | Mans portrait, smiling",
                 "Old Womans portrait | Old Womans portrait, smiling",
                "Old Mans portrait | Old Mans portrait, smiling",
                "Beautiful, snowy Tokyo city is bustling. The camera moves through the bustling city street, following several people enjoying the beautiful snowy weather and shopping at nearby stalls",
                "At dusk, a car is driving on the highway, with the rearview mirror reflecting a colorful sunset and serene scenery",
                "Extreme close-up of chicken and green pepper kebabs grilling on a barbeque with flames. Shallow focus and light smoke. vivid colours",
                "A side profile shot of a woman with fireworks exploding in the distance beyond her",
                "A boat sailing leisurely along the Seine River with the Eiffel Tower in background, black and white",
                "A cat waking up its sleeping owner demanding breakfast",
                "A drone camera circles around a beautiful historic church built on a rocky outcropping along the Amalfi Coast, the view showcases historic and magnificent architectural details and tiered pathways and patios",
                "Drone view of waves crashing against the rugged cliffs along Big Sur’s garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore",
                "An aerial shot of a lighthouse standing tall on a rocky cliff, its beacon cutting through the early dawn, waves crash against the rocks below",
                "Churning ocean waves at night with a lighthouse on the coast create an intense and somewhat foreboding atmosphere. The scene is set under an overcast sky, with the ocean’s dark waters illuminated by natural light",
                "A tsunami coming through an alley in Bulgaria, dynamic movement",
                "A massive explosion on the surface of the earth",
                "A campfire burning with flames and embers, gradually increasing in size and intensity before dying down towards the end",
                "Two hikers make their way up a mountain, taking in the fresh air and beautiful views",
                "A stunning sight as a steam train leaves the bridge, traveling over the arch-covered viaduct. The landscape is dotted with lush greenery and rocky mountains. The sky is blue and the sun is shining.",
                "A car driving on the road",
                "FPV flying over the Great Wall",
                "People strolling through the trees of a small suburban park on an island in the river on an ordinary Sunday",
                "Renaissance-style portrait of an astronaut in space, detailed starry background, reflective helmet." ,
                "Surrealist painting of a floating island with giant clock gears, populated with mythical creatures." ,
                "Abstract painting representing the sound of jazz music, using vibrant colors and erratic shapes." ,
                "Pop Art painting of a modern smartphone with classic art pieces appearing on the screen." ,
                "Baroque-style battle scene with futuristic robots and a golden palace in the background." ,
                "Cubist painting of a bustling city market with different perspectives of people and stalls." ,
                "Romantic painting of a ship sailing in a stormy sea, with dramatic lighting and powerful waves." ,
                "Art Nouveau painting of a female botanist surrounded by exotic plants in a greenhouse." ,
                "Gothic painting of an ancient castle at night, with a full moon, gargoyles, and shadows." ,
                "Black and white street photography of a rainy night in New York, reflections on wet pavement." ,
                "High-fashion photography in an abandoned industrial warehouse, with dramatic lighting and edgy outfits." ,
                "Macro photography of dewdrops on a spiderweb, with morning sunlight creating rainbows." ,
                "Aerial photography of a winding river through autumn forests, with vibrant red and orange foliage." ,
                "Urban portrait of a skateboarder in mid-jump, graffiti walls background, high shutter speed." ,
                "Underwater photography of a coral reef, with diverse marine life and a scuba diver for scale." ,
                "Vintage-style travel photography of a train station in Europe, with passengers and old luggage." ,
                "Long-exposure night photography of a starry sky over a mountain range, with light trails." ,
                "Documentary-style photography of a bustling marketplace in Marrakech, with spices and textiles." ,
                "Food photography of a gourmet meal, with a shallow depth of field, elegant plating, and soft lighting." ,
                "Cyberpunk cityscape with towering skyscrapers, neon signs, and flying cars." ,
                "Fantasy illustration of a dragon perched on a castle, with a stormy sky and lightning in the background." ,
                "Digital painting of an astronaut floating in space, with a reflection of Earth in the helmet visor." ,
                "Concept art for a post-apocalyptic world with ruins, overgrown vegetation, and a lone survivor." ,
                "Pixel art of an 8-bit video game character running through a retro platformer level." ,
                "Isometric digital art of a medieval village with thatched roofs, a market square, and townsfolk." ,
                "Digital illustration in manga style of a samurai warrior in a duel against a mystical creature." ,
                "A minimalistic digital artwork of an abstract geometric pattern with a harmonious color palette." ,
                "Sci-fi digital painting of an alien landscape with otherworldly plants, strange creatures, and distant planets." ,
                "Steampunk digital art of an inventor’s workshop, with intricate machines, gears, and steam engines."
            };

            OutputHelpers.WriteConsole($"Loading Pipeline `{pipeline.PipelineType} - {pipeline.Name}`...", ConsoleColor.Cyan);

            //    var videoInput = await OnnxVideo.FromFileAsync("C:\\Users\\Deven\\Pictures\\2.gif", 9.8f, width: 512, height: 512);
            foreach (var prompt in prompts)
            {
                var promptIndex = prompts.IndexOf(prompt);
                var generateOptions = new GenerateOptions
                {
                    MotionContextOverlap = 0,
                    MotionNoiseContext = 16,
                    MotionFrames = motionFrames,
                    InputFrameRate = 24,
                    OutputFrameRate = 24,
                    Diffuser = DiffuserType.TextToVideo,
                    Prompt = prompt,
                    Prompts = prompt.Split("|", StringSplitOptions.TrimEntries).ToList(),
                    NegativePrompt = "painting, drawing, sketches, illustration, cartoon, crayon"
                };


                foreach (var aspect in new[] { new { Width = 512, Height = 512 } })
                //foreach (var aspect in new[] { new { Width = 512, Height = 512 }, new { Width = 704, Height = 448 }, new { Width = 448, Height = 704 } })
                {

                    generateOptions.SchedulerOptions = pipeline.DefaultSchedulerOptions with
                    {
                        Seed = developmentSeed,
                        SchedulerType = SchedulerType.Locomotion,
                        BetaSchedule = BetaScheduleType.Linear,
                        InferenceSteps = 8,
                        TimestepSpacing = TimestepSpacingType.Trailing,
                        StepsOffset = 1,
                        //UseKarrasSigmas = true,

                        Strength = 0.8f,

                        Height = aspect.Height,
                        Width = aspect.Width,
                        GuidanceScale = guidanceScale,

                    };

                    var timestamp = Stopwatch.GetTimestamp();
                    OutputHelpers.WriteConsole($"Generating {generateOptions.SchedulerOptions.SchedulerType} Image...", ConsoleColor.Green);

                    // Run pipeline
                    var video = await pipeline.GenerateVideoAsync(generateOptions, progressCallback: OutputHelpers.ProgressCallback);

                    // Save Image File
                    var videoFilename = Path.Combine(_outputDirectory, $"{promptIndex}-{generateOptions.SchedulerOptions.Width}x{generateOptions.SchedulerOptions.Height}-{generateOptions.SchedulerOptions.SchedulerType}-{generateOptions.MotionFrames}-{generateOptions.SchedulerOptions.GuidanceScale:F2}-{generateOptions.SchedulerOptions.Seed}-{postfix}.mp4");

                    video.NormalizeBrightness();
                    await video.SaveAsync(videoFilename);

                    OutputHelpers.WriteConsole($"Image Created: {Path.GetFileName(videoFilename)}", ConsoleColor.Green);
                    OutputHelpers.WriteConsole($"Elapsed: {Stopwatch.GetElapsedTime(timestamp)}ms", ConsoleColor.Yellow);
                }
            }

            // Unload pipeline
            await pipeline.UnloadAsync();
            OutputHelpers.WriteConsole($"Done - Elapsed: {Stopwatch.GetElapsedTime(totalElapsed)}", ConsoleColor.Yellow);
        }


    }
}
