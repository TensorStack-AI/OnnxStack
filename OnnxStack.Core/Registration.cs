using Microsoft.Extensions.DependencyInjection;
using OnnxStack.Common.Config;
using OnnxStack.Core.Config;
using OnnxStack.Core.Services;

namespace OnnxStack.Core
{
    /// <summary>
    /// .NET Core Service and Dependancy Injection registration helpers
    /// </summary>
    public static class Registration
    {
        /// <summary>
        /// Register OnnxStack services
        /// </summary>
        /// <param name="serviceCollection">The service collection.</param>
        public static void AddOnnxStack(this IServiceCollection serviceCollection)
        {
            serviceCollection.AddSingleton(ConfigManager.LoadConfiguration());
            serviceCollection.AddSingleton<IOnnxModelService, OnnxModelService>();
            serviceCollection.AddSingleton<IOnnxModelAdaptaterService, OnnxModelAdaptaterService>();
        }


        /// <summary>
        /// Register a custom IConfigSection section that is in the appsettings.json
        /// </summary>
        /// <typeparam name="T">The custom IConfigSection class type, NOTE: json section name MUST match class name</typeparam>
        /// <param name="serviceCollection">The service collection.</param>
        public static void AddOnnxStackConfig<T>(this IServiceCollection serviceCollection) 
            where T : class, IConfigSection
        {
            serviceCollection.AddSingleton(ConfigManager.LoadConfiguration<T>());
        }
    }
}
