using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Pipelines;
using System.Diagnostics;

namespace OnnxStack.Console.Runner
{
    public sealed class ModelSamples : IExampleRunner
    {
        private readonly string _outputDirectory;

        public ModelSamples()
        {
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(ModelSamples));
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 2;

        public string Name => "Model Samples";

        public string Description => "Generate sample images";

        public async Task RunAsync()
        {
            // Execution provider
            var provider = Providers.DirectML(0);

            // Create Pipeline
            // var pipeline = StableDiffusionPipeline.CreatePipeline("M:\\Models\\stable-diffusion-v1-5-onnx");
            // var pipeline = StableDiffusionXLPipeline.CreatePipeline("M:\\Models\\stable-diffusion-xl-base-1.0-onnx");
            // var pipeline = StableDiffusion3Pipeline.CreatePipeline("M:\\Models\\stable-diffusion-3-medium-onnx");
            // var pipeline = FluxPipeline.CreatePipeline("M:\\Models\\Flux_schnell-f16-onnx", memoryMode: MemoryModeType.Minimum);
            // var pipeline = LatentConsistencyPipeline.CreatePipeline("M:\\Models\\LCM_Dreamshaper_v7-onnx");
            // var pipeline = LatentConsistencyXLPipeline.CreatePipeline("M:\\Models\\Latent-Consistency-xl-Olive-Onnx");
            // var pipeline = CogVideoPipeline.CreatePipeline("M:\\BaseModels\\CogVideoX-2b\\_Test", memoryMode: MemoryModeType.Minimum);
            // var pipeline = HunyuanPipeline.CreatePipeline("M:\\Models\\HunyuanDiT-onnx", memoryMode: MemoryModeType.Maximum);
            // var pipeline = StableDiffusionPipeline.CreatePipeline("M:\\Models\\stable-diffusion-instruct-pix2pix-onnx", ModelType.Instruct);
            var pipeline = LocomotionPipeline.CreatePipeline(provider, "M:\\Models\\Locomotion-v1-LCM", 16);

            var index = 0;
            var sampleSeed = 1445933838;
            var timestamp = Stopwatch.GetTimestamp();
            var outputDir = Directory.CreateDirectory(Path.Combine(_outputDirectory, pipeline.Name));
            foreach (var prompt in SamplePrompts)
            {
                // Options
                var generateOptions = new GenerateOptions
                {
                    MotionFrames = 16,
                    MotionNoiseContext = 16,
                    MotionContextOverlap = 1,

                    InputFrameRate = 24,
                    OutputFrameRate = 24,

                    FrameResample = false,
                    Prompt = prompt + PromptAddon,
                    NegativePrompt = NegativePrompt,
                    Diffuser = DiffuserType.TextToVideo,
                    SchedulerOptions = pipeline.DefaultSchedulerOptions with
                    {
                        Width = 512,
                        Height = 512,

                        //Width = 704,
                        //Height = 448,
                        Seed = sampleSeed,
                        InferenceSteps = 8,
                        GuidanceScale = 0f,
                        SchedulerType = SchedulerType.Locomotion,
                      

                        //  BetaEnd = 0.015f
                    }
                };

                OutputHelpers.WriteConsole($"[{++index}/{SamplePrompts.Count}] {prompt}", ConsoleColor.Yellow);

                // Run pipeline
                var result = await pipeline.GenerateVideoAsync(generateOptions, progressCallback: OutputHelpers.ProgressCallback);

                // Save Image File
                await result.SaveAsync(Path.Combine(outputDir.FullName, $"{index}.mp4"));
            }

            var elapsed = Stopwatch.GetElapsedTime(timestamp);
            OutputHelpers.WriteConsole($"Elapsed: {elapsed}, Images/Min: {SamplePrompts.Count / elapsed.TotalMinutes:F2}", ConsoleColor.Green);

            //Unload
            await pipeline.UnloadAsync();
        }

        private static string PromptAddon = ", hyperrealism, photorealistic, 4k";
        private static string NegativePrompt = "painting, drawing, sketches, monochrome, grayscale, crayon, graphite, abstract, watermark, signature";

        private static List<string> SamplePrompts = [
            "Create an image of a serene sunset over a calm lake, with the colors reflecting in the water",
            "Illustrate a fierce warrior woman in ornate armor, wielding a glowing sword and standing atop a mountain peak",
            "Create an image of a wise old wizard with a long white beard, holding a staff topped with a crystal and casting a spell",
            "Depict a graceful elven archer, with pointed ears and a bow drawn, ready to release an arrow in an enchanted forest",
            "Design a powerful sorceress, surrounded by swirling magical energy, with runes glowing on her skin",
            "Generate a scene of a noble knight in shining armor, riding a majestic horse and carrying a banner into battle",
            "Visualize a mysterious rogue woman, cloaked in shadows with daggers at the ready, sneaking through a moonlit alley",
            "Craft an image of a regal queen in a flowing gown, wearing a crown and holding a scepter, standing in a grand hall",
            "Illustrate a fierce barbarian man, with muscles rippling and wielding a massive axe, roaring in triumph",
            "Create a scene of a beautiful fairy woman with delicate wings, hovering over a bed of flowers and sprinkling magic dust",
            "Depict a stoic dwarf warrior, with a braided beard and a massive hammer, standing guard in a mountain stronghold",
            "Design an enchanting mermaid with shimmering scales, sitting on a rock by the ocean, singing to the moon",
            "Generate an image of a dark sorcerer, with a menacing aura and dark robes, conjuring dark magic in a cavern",
            "Visualize a brave adventurer woman, with a map and a lantern, exploring ancient ruins in search of treasure",
            "Craft an image of a gallant prince, with a charming smile and a sword at his side, ready to embark on a quest",
            "Illustrate a fierce dragon rider, with a majestic dragon beneath her, soaring through the skies",
            "Create a scene of a mystical shaman, with animal skins and feathers, performing a ritual in a sacred grove",
            "Depict a heroic paladin man, with a shield and a glowing sword, standing against a horde of demons",
            "Design an ethereal spirit woman, with translucent skin and flowing hair, floating gracefully through an ancient forest",
            "Generate an image of a cunning thief man, with a hood and a smirk, slipping away with a bag of stolen jewels",
            "Visualize a radiant goddess, with a celestial aura and divine power, standing atop a mountain with the sun rising behind her",
            "Illustrate a cheerful barista preparing a cup of coffee behind the counter of a cozy cafe.",
            "Create an image of a young woman jogging through a park, with headphones on and a determined look.",
            "Depict a businessman hailing a cab on a busy city street, with skyscrapers in the background.",
            "Design a scene of a grandmother knitting a sweater in a comfortable living room, with a cat curled up beside her.",
            "Generate an image of a father teaching his child how to ride a bicycle in a suburban neighborhood.",
            "Visualize a group of friends enjoying a picnic in a sunny park, with a spread of food and drinks on a blanket.",
            "Craft an image of a teacher standing at a chalkboard, explaining a lesson to a classroom full of attentive students.",
            "Illustrate a young woman reading a book in a library, surrounded by shelves filled with books.",
            "Create a scene of a chef in a bustling kitchen, chopping vegetables and preparing a gourmet meal.",
            "Depict a young couple walking hand-in-hand along a beach at sunset, with waves lapping at their feet.",
            "Design an image of a mechanic working on a car in a garage, with tools and spare parts scattered around.",
            "Generate a scene of a mail carrier delivering letters and packages in a residential neighborhood.",
            "Visualize a group of children playing soccer on a grassy field, with joyful expressions and energetic movements.",
            "Craft an image of a nurse taking care of a patient in a hospital room, with medical equipment and a warm smile.",
            "Illustrate a young man playing guitar on a street corner, with a hat for tips and a small audience gathered around.",
            "Create a scene of a mother reading a bedtime story to her child, with a cozy bed and soft lighting.",
            "Depict a painter working on a canvas in an art studio, surrounded by paint supplies and unfinished works.",
            "Design an image of a young woman shopping at a farmers market, selecting fresh fruits and vegetables.",
            "Generate a scene of a construction worker operating heavy machinery at a building site, with cranes and scaffolding.",
            "Visualize a family enjoying a day at the beach, building sandcastles and splashing in the waves.",
            "Illustrate a sleek sports car speeding along a coastal highway, with the ocean on one side and mountains on the other.",
            "Create an image of a luxury yacht anchored in a serene bay, with crystal-clear water and a sunset in the background.",
            "Depict a vintage biplane soaring through the sky, with puffy white clouds and a bright blue sky.",
            "Design a scene of a futuristic electric car charging at a modern station, with a cityscape in the background.",
            "Generate an image of a classic steam locomotive chugging along a countryside railway, with plumes of steam rising.",
            "Visualize a majestic sailing ship navigating through rough seas, with sails billowing and waves crashing.",
            "Craft an image of a modern commercial airplane taking off from a busy airport runway, with terminal buildings visible.",
            "Illustrate a rugged off-road vehicle climbing a steep mountain trail, with rocky terrain and a vast landscape below.",
            "Create a scene of a colorful hot air balloon drifting over a picturesque valley, with other balloons in the distance.",
            "Depict a luxurious private jet flying above the clouds, with the sun setting on the horizon.",
            "Design an image of a classic muscle car parked in front of a retro diner, with neon lights and a nostalgic atmosphere.",
            "Generate a scene of a high-speed bullet train gliding through a futuristic cityscape, with skyscrapers and neon signs.",
            "Visualize a traditional wooden fishing boat anchored in a quiet harbor, with nets and fishing gear on board.",
            "Craft an image of a sleek motorbike racing down a winding road, with mountains and forests in the background.",
            "Illustrate a vintage convertible car driving along a scenic coastal road, with the top down and the wind blowing.",
            "Create a scene of a powerful jet fighter flying in formation, with a dramatic sky and contrails behind them.",
            "Depict a spacious RV parked at a campsite, with a family setting up a picnic table and enjoying the outdoors.",
            "Design an image of a luxurious cruise ship sailing across the ocean, with passengers relaxing on deck chairs.",
            "Generate a scene of a futuristic hovercraft gliding over water, with a futuristic city in the distance.",
            "Visualize a classic American pickup truck parked by a rustic barn, with fields and a sunset in the background.",
            "Generate a scene of a person rock climbing on a rugged cliff face, with ropes and safety gear.",
            "Visualize a person exploring an ancient ruin, with moss-covered stones and hidden chambers.",
            "Craft an image of a person riding a horse through a peaceful countryside, with fields and forests stretching to the horizon.",
            "Design a scene of a person practicing photography in an urban setting, capturing street scenes and architecture.",
            "Illustrate a group of friends camping in the wilderness, with tents pitched and a campfire burning.",
            "Depict a person playing chess in a park, with a chessboard set up on a picnic table.",
            "Create an image of a person painting a mural on a city wall, with vibrant colors and intricate designs.",
            "Illustrate a classic car race with vintage cars zooming along a winding track.",
            "Depict a colorful array of toys scattered across a child's playroom, from plush animals to building blocks.",
            "Create an image of a mouthwatering gourmet burger with layers of juicy beef, cheese, and fresh vegetables.",
            "Generate a 3D render of a futuristic cityscape with sleek skyscrapers reaching towards the sky and flying cars zipping between them.",
            "Visualize a cartoonish scene of anthropomorphic animals having a picnic in a sunny meadow, with food and laughter abound.",
            "Craft an image of a galaxy far, far away, with swirling nebulae and distant stars painting the cosmic canvas.",
            "Design a scene of a bustling food market with vendors selling exotic fruits, spices, and street food delicacies.",
            "Illustrate a collection of colorful balloons drifting in a bright blue sky, carried by a gentle breeze.",
            "Depict a cozy cafe corner with steaming cups of coffee, freshly baked pastries, and a stack of books waiting to be read.",
            "Create an image of a retro arcade filled with flashing lights, beeping sounds, and rows of arcade cabinets.",
            "Generate a 3D render of a fantastical underwater kingdom with coral reefs, sea creatures, and sunken treasures.",
            "Visualize a whimsical scene of a toy workshop bustling with activity, as elves craft toys for children around the world.",
            "Craft an image of a vibrant fruit market with piles of ripe oranges, bananas, and watermelons stacked on wooden crates.",
            "Design a cartoonish illustration of a mischievous cat getting into trouble, surrounded by torn paper and spilled ink bottles.",
            "Illustrate a picnic blanket spread out under a shady tree, with sandwiches, fruit, and a thermos of lemonade.",
            "Depict a sleek sports car speeding along a winding mountain road, with the wind in its driver's hair.",
            "Create an image of a festive dinner table set for a holiday feast, with turkey, mashed potatoes, and cranberry sauce.",
            "Generate a 3D render of a space station orbiting a distant planet, with astronauts floating weightlessly outside.",
            "Visualize a cartoonish scene of a wizard's cluttered study, with spellbooks, potion bottles, and magical artifacts.",
            "Craft an image of a bustling street food market with vendors cooking up sizzling kebabs, noodles, and dumplings.",
            "Design a scene of a child's bedroom filled with toys, books, and stuffed animals, with a cozy bed in the corner.",
            "Illustrate a vintage diner with neon signs, checkered floors, and chrome barstools lined up along the counter.",
            "Depict a toy train winding its way through a miniature town, past tiny houses and trees.",
            "Create an image of a colorful bouquet of flowers in a vase, brightening up a room with their beauty.",
            "Generate a 3D render of a futuristic robot factory with assembly lines, robotic arms, and conveyor belts.",
            "Visualize a cartoonish scene of a superhero battling a giant monster in a bustling city, with buildings crumbling around them.",
            "Craft an image of a cozy kitchen with pots bubbling on the stove, fresh bread baking in the oven, and cookies cooling on a rack.",
            "Design a scene of a medieval castle with turrets, battlements, and a drawbridge lowering over a moat.",
            "Illustrate a bustling night market with stalls selling street food, souvenirs, and handmade crafts.",
            "Depict a toy chest overflowing with dolls, action figures, toy cars, and stuffed animals.",
            "Create an image of a gourmet pizza fresh from the oven, with bubbling cheese, savory toppings, and a crispy crust.",
            "Generate a 3D render of a futuristic cyberpunk city with towering skyscrapers, neon lights, and flying vehicles.",
            "Visualize a whimsical scene of a magical forest with talking animals, glowing mushrooms, and hidden fairy homes.",
            "Craft an image of a cozy fireplace with crackling flames, flickering candles, and a pile of cozy blankets nearby.",
            "Design a cartoonish illustration of a group of adventurers setting out on a perilous journey, with swords, maps, and backpacks.",
            "Illustrate a toy chest overflowing with Lego bricks, ready for endless hours of building and creativity.",
            "Depict a tropical beach paradise with palm trees, white sand, and crystal-clear water lapping at the shore.",
            "Create an image of a bustling city street with cars, buses, and pedestrians bustling about their day.",
            "Generate a 3D render of a sleek sci-fi spaceship soaring through the stars, with engines blazing and stars streaking past.",
            "Visualize a cartoonish scene of a pirate ship sailing the high seas, with Jolly Roger flags and a crew of colorful characters.",
            "Craft an image of a cozy mountain cabin surrounded by snowy peaks, with smoke rising from the chimney."
        ];
    }
}
