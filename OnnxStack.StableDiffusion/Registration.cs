using Microsoft.Extensions.DependencyInjection;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Services;

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
            serviceCollection.AddSingleton<IImageService, ImageService>();
            serviceCollection.AddSingleton<IInferenceService, InferenceService>();
            serviceCollection.AddSingleton<IStableDiffusionService, StableDiffusionService>();
        }
    }
}
