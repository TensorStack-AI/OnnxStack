using Microsoft.Extensions.DependencyInjection;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.ImageUpscaler.Config;
using OnnxStack.ImageUpscaler.Services;

namespace OnnxStack.ImageUpscaler
{
    public static class Registration
    {
        /// <summary>
        /// Register OnnxStack ImageUpscaler services
        /// </summary>
        /// <param name="serviceCollection">The service collection.</param>
        public static void AddOnnxStackImageUpscaler(this IServiceCollection serviceCollection)
        {
            serviceCollection.AddOnnxStack();
            serviceCollection.RegisterServices();
            serviceCollection.AddSingleton(ConfigManager.LoadConfiguration<ImageUpscalerConfig>());
        }


        /// <summary>
        /// Register OnnxStack ImageUpscaler services, AddOnnxStack() must be called before
        /// </summary>
        /// <param name="serviceCollection">The service collection.</param>
        /// <param name="configuration">The configuration.</param>
        public static void AddOnnxStackImageUpscaler(this IServiceCollection serviceCollection, ImageUpscalerConfig configuration)
        {
            serviceCollection.RegisterServices();
            serviceCollection.AddSingleton(configuration);
        }


        private static void RegisterServices(this IServiceCollection serviceCollection)
        {
            serviceCollection.AddSingleton<IImageService, ImageService>();
            serviceCollection.AddSingleton<IUpscaleService, UpscaleService>();
        }
    }
}