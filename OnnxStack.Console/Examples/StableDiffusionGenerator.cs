using OnnxStack.Core;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using SixLabors.ImageSharp;
using System.Collections.ObjectModel;

namespace OnnxStack.Console.Runner
{
    public sealed class StableDiffusionGenerator : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly IStableDiffusionService _stableDiffusionService;
        private readonly ReadOnlyDictionary<string, string> _generationPrompts;

        public StableDiffusionGenerator(IStableDiffusionService stableDiffusionService)
        {
            _stableDiffusionService = stableDiffusionService;
            _generationPrompts = GeneratePrompts();
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(StableDiffusionGenerator));
        }

        public string Name => "Stable Diffusion Generator";

        public string Description => "Generator images from fixed prompts using all Scheduler types";

        /// <summary>
        /// Stable Diffusion Demo
        /// </summary>
        public async Task RunAsync()
        {
            Directory.CreateDirectory(_outputDirectory);

            var seed = Random.Shared.Next();
            foreach (var model in _stableDiffusionService.Models)
            {
                OutputHelpers.WriteConsole($"Loading Model `{model.Name}`...", ConsoleColor.Green);
                await _stableDiffusionService.LoadModel(model);

                foreach (var generationPrompt in _generationPrompts)
                {
                    var promptOptions = new PromptOptions
                    {
                        Prompt = generationPrompt.Value
                    };

                    var schedulerOptions = new SchedulerOptions
                    {
                        Seed = Random.Shared.Next()
                    };
                    foreach (var schedulerType in Helpers.GetPipelineSchedulers(model.PipelineType))
                    {
                        promptOptions.SchedulerType = schedulerType;
                        OutputHelpers.WriteConsole($"Generating {schedulerType} Image...", ConsoleColor.Green);
                        await GenerateImage(model, promptOptions, schedulerOptions, generationPrompt.Key);
                    }
                }

                OutputHelpers.WriteConsole($"Unloading Model `{model.Name}`...", ConsoleColor.Green);
                await _stableDiffusionService.UnloadModel(model);
            }
            OutputHelpers.WriteConsole("Complete :)", ConsoleColor.DarkMagenta);
            OutputHelpers.ReadConsole(ConsoleColor.Gray);
        }

        private async Task<bool> GenerateImage(ModelOptions model, PromptOptions prompt, SchedulerOptions options, string key)
        {
            var outputFilename = Path.Combine(_outputDirectory, $"{options.Seed}_{prompt.SchedulerType}_{key}.png");
            var result = await _stableDiffusionService.GenerateAsImageAsync(model, prompt, options);
            if (result == null)
                return false;

            await result.SaveAsPngAsync(outputFilename);
            OutputHelpers.WriteConsole($"{prompt.SchedulerType} Image Created: {Path.GetFileName(outputFilename)}", ConsoleColor.Green);
            return true;
        }

        private ReadOnlyDictionary<string, string> GeneratePrompts()
        {
            return new Dictionary<string, string>
            {
                {"The Renaissance Astrounaut","Renaissance-style portrait of an astronaut in space, detailed starry background, reflective helmet." },
                {"The Surreal Floating Island","Surrealist painting of a floating island with giant clock gears, populated with mythical creatures." },
                {"Jazz in Abstract Colors","Abstract painting representing the sound of jazz music, using vibrant colors and erratic shapes." },
                {"The Confluence of Pop Art","Pop Art painting of a modern smartphone with classic art pieces appearing on the screen." },
                {"The Robotic Baroque Battle","Baroque-style battle scene with futuristic robots and a golden palace in the background." },
                {"Cubist Bustling Market","Cubist painting of a bustling city market with different perspectives of people and stalls." },
                {"The Romantic Stormy Voyage","Romantic painting of a ship sailing in a stormy sea, with dramatic lighting and powerful waves." },
                {"The Botanist in Art Nouveau","Art Nouveau painting of a female botanist surrounded by exotic plants in a greenhouse." },
                {"The Gothic Moonlit Castle","Gothic painting of an ancient castle at night, with a full moon, gargoyles, and shadows." },
                {"Rainy New York Nights","Black and white street photography of a rainy night in New York, reflections on wet pavement." },
                {"The Fashion of Abandoned Places","High-fashion photography in an abandoned industrial warehouse, with dramatic lighting and edgy outfits." },
                {"Rainbow Dewdrops","Macro photography of dewdrops on a spiderweb, with morning sunlight creating rainbows." },
                {"Aerial Autumn River","Aerial photography of a winding river through autumn forests, with vibrant red and orange foliage." },
                {"Skateboarders Urban Flight","Urban portrait of a skateboarder in mid-jump, graffiti walls background, high shutter speed." },
                {"Dive into Coral Reefs","Underwater photography of a coral reef, with diverse marine life and a scuba diver for scale." },
                {"Vintage European Transit","Vintage-style travel photography of a train station in Europe, with passengers and old luggage." },
                {"Star Trails over Mountains","Long-exposure night photography of a starry sky over a mountain range, with light trails." },
                {"Marketplace Colors of Marrakech","Documentary-style photography of a bustling marketplace in Marrakech, with spices and textiles." },
                {"Elegance on a Plate","Food photography of a gourmet meal, with a shallow depth of field, elegant plating, and soft lighting." },
                {"Neon-Soaked Cyberpunk City","Cyberpunk cityscape with towering skyscrapers, neon signs, and flying cars." },
                {"Dragon’s Stormy Perch","Fantasy illustration of a dragon perched on a castle, with a stormy sky and lightning in the background." },
                {"Reflections of Earth","Digital painting of an astronaut floating in space, with a reflection of Earth in the helmet visor." },
                {"After The Fall","Concept art for a post-apocalyptic world with ruins, overgrown vegetation, and a lone survivor." },
                {"Retro Gaming Nostalgia","Pixel art of an 8-bit video game character running through a retro platformer level." },
                {"Medieval Village Life","Isometric digital art of a medieval village with thatched roofs, a market square, and townsfolk." },
                {"Samurai and the Mystical","Digital illustration in manga style of a samurai warrior in a duel against a mystical creature." },
                {"Minimalistic Geometry","A minimalistic digital artwork of an abstract geometric pattern with a harmonious color palette." },
                {"Alien Flora and Fauna","Sci-fi digital painting of an alien landscape with otherworldly plants, strange creatures, and distant planets." },
                {"The Inventors Steampunk Workshop","Steampunk digital art of an inventor’s workshop, with intricate machines, gears, and steam engines." }
            }.AsReadOnly();
        }
    }
}
