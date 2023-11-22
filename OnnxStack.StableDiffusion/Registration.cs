using Microsoft.Extensions.DependencyInjection;
using OnnxStack.Core.Config;
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
            ConfigureLibraries();
            serviceCollection.AddOnnxStack();
            serviceCollection.AddSingleton(ConfigManager.LoadConfiguration<StableDiffusionConfig>(nameof(OnnxStackConfig)));

            // Services
            serviceCollection.AddSingleton<IPromptService, PromptService>();
            serviceCollection.AddSingleton<IStableDiffusionService, StableDiffusionService>();

            //Pipelines
            serviceCollection.AddSingleton<IPipeline, StableDiffusionPipeline>();
            serviceCollection.AddSingleton<IPipeline, LatentConsistencyPipeline>();

            //StableDiffusion
            serviceCollection.AddSingleton<IDiffuser, StableDiffusion.Diffusers.StableDiffusion.TextDiffuser>();
            serviceCollection.AddSingleton<IDiffuser, StableDiffusion.Diffusers.StableDiffusion.ImageDiffuser>();
            serviceCollection.AddSingleton<IDiffuser, StableDiffusion.Diffusers.StableDiffusion.InpaintDiffuser>();
            serviceCollection.AddSingleton<IDiffuser, StableDiffusion.Diffusers.StableDiffusion.InpaintLegacyDiffuser>();

            //LatentConsistency
            serviceCollection.AddSingleton<IDiffuser, StableDiffusion.Diffusers.LatentConsistency.TextDiffuser>();
            serviceCollection.AddSingleton<IDiffuser, StableDiffusion.Diffusers.LatentConsistency.ImageDiffuser>();
            serviceCollection.AddSingleton<IDiffuser, StableDiffusion.Diffusers.LatentConsistency.InpaintLegacyDiffuser>();
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
    }
}
