using Microsoft.Extensions.DependencyInjection;
using OnnxStack.Core.Config;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Pipelines;
using OnnxStack.StableDiffusion.Services;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Memory;

namespace OnnxStack.Core
{
    /// <summary>
    /// .NET Core Service and Dependancy Injection registration helpers
    /// </summary>
    public static class Registration
    {
        /// <summary>
        /// Register OnnxStack StableDiffusion services
        /// </summary>
        /// <param name="serviceCollection">The service collection.</param>
        public static void AddOnnxStackStableDiffusion(this IServiceCollection serviceCollection)
        {
            serviceCollection.AddOnnxStack();
            serviceCollection.RegisterServices();
            serviceCollection.AddSingleton(TryLoadAppSettings());
        }


        /// <summary>
        /// Register OnnxStack StableDiffusion services, AddOnnxStack() must be called before
        /// </summary>
        /// <param name="serviceCollection">The service collection.</param>
        /// <param name="configuration">The configuration.</param>
        public static void AddOnnxStackStableDiffusion(this IServiceCollection serviceCollection, StableDiffusionConfig configuration)
        {
            serviceCollection.RegisterServices();
            serviceCollection.AddSingleton(configuration);
        }


        private static void RegisterServices(this IServiceCollection serviceCollection)
        {
            ConfigureLibraries();

            // Services
            serviceCollection.AddSingleton<IVideoService, VideoService>();
            serviceCollection.AddSingleton<IPromptService, PromptService>();
            serviceCollection.AddSingleton<IStableDiffusionService, StableDiffusionService>();

            //Pipelines
            serviceCollection.AddSingleton<IPipeline, StableDiffusionPipeline>();
            serviceCollection.AddSingleton<IPipeline, StableDiffusionXLPipeline>();
            serviceCollection.AddSingleton<IPipeline, LatentConsistencyPipeline>();
            serviceCollection.AddSingleton<IPipeline, LatentConsistencyXLPipeline>();
            serviceCollection.AddSingleton<IPipeline, InstaFlowPipeline>();
            serviceCollection.AddSingleton<IPipeline, ControlNetPipeline>();

            //StableDiffusion
            serviceCollection.AddSingleton<IDiffuser, StableDiffusion.Diffusers.StableDiffusion.TextDiffuser>();
            serviceCollection.AddSingleton<IDiffuser, StableDiffusion.Diffusers.StableDiffusion.ImageDiffuser>();
            serviceCollection.AddSingleton<IDiffuser, StableDiffusion.Diffusers.StableDiffusion.InpaintDiffuser>();
            serviceCollection.AddSingleton<IDiffuser, StableDiffusion.Diffusers.StableDiffusion.InpaintLegacyDiffuser>();

            //StableDiffusionXL
            serviceCollection.AddSingleton<IDiffuser, StableDiffusion.Diffusers.StableDiffusionXL.TextDiffuser>();
            serviceCollection.AddSingleton<IDiffuser, StableDiffusion.Diffusers.StableDiffusionXL.ImageDiffuser>();
            serviceCollection.AddSingleton<IDiffuser, StableDiffusion.Diffusers.StableDiffusionXL.InpaintLegacyDiffuser>();

            //LatentConsistency
            serviceCollection.AddSingleton<IDiffuser, StableDiffusion.Diffusers.LatentConsistency.TextDiffuser>();
            serviceCollection.AddSingleton<IDiffuser, StableDiffusion.Diffusers.LatentConsistency.ImageDiffuser>();
            serviceCollection.AddSingleton<IDiffuser, StableDiffusion.Diffusers.LatentConsistency.InpaintLegacyDiffuser>();

            //LatentConsistencyXL
            serviceCollection.AddSingleton<IDiffuser, StableDiffusion.Diffusers.LatentConsistencyXL.TextDiffuser>();
            serviceCollection.AddSingleton<IDiffuser, StableDiffusion.Diffusers.LatentConsistencyXL.ImageDiffuser>();
            serviceCollection.AddSingleton<IDiffuser, StableDiffusion.Diffusers.LatentConsistencyXL.InpaintLegacyDiffuser>();

            //InstaFlow
            serviceCollection.AddSingleton<IDiffuser, StableDiffusion.Diffusers.InstaFlow.TextDiffuser>();

            //ControlNet
            serviceCollection.AddSingleton<IDiffuser, StableDiffusion.Diffusers.ControlNet.ImageDiffuser>();
        }


        /// <summary>
        /// Configures any 3rd party libraries.
        /// </summary>
        private static void ConfigureLibraries()
        {
            // Create a 100MB image buffer pool
            Configuration.Default.PreferContiguousImageBuffers = true;
            Configuration.Default.MemoryAllocator = MemoryAllocator.Create(new MemoryAllocatorOptions
            {
                MaximumPoolSizeMegabytes = 100,
            });
        }


        /// <summary>
        /// Try load StableDiffusionConfig from application settings.
        /// </summary>
        /// <returns></returns>
        private static StableDiffusionConfig TryLoadAppSettings()
        {
            try
            {
                return ConfigManager.LoadConfiguration<StableDiffusionConfig>();
            }
            catch
            {
                return new StableDiffusionConfig();
            }
        }
    }
}
