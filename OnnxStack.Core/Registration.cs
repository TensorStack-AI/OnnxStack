using Microsoft.Extensions.DependencyInjection;
using OnnxStack.Common.Config;
using OnnxStack.Core.Config;

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
            serviceCollection.AddSingleton(TryLoadAppSettings());
        }


        /// <summary>
        /// Register OnnxStack services
        /// </summary>
        /// <param name="serviceCollection">The service collection.</param>
        /// <param name="configuration">The configuration.</param>
        public static void AddOnnxStack(this IServiceCollection serviceCollection, OnnxStackConfig configuration)
        {
            serviceCollection.AddSingleton(configuration);
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


        /// <summary>
        /// Try load OnnxStackConfig from application settings if it exists.
        /// </summary>
        /// <returns></returns>
        private static OnnxStackConfig TryLoadAppSettings()
        {
            try
            {
                return ConfigManager.LoadConfiguration<OnnxStackConfig>();
            }
            catch
            {
                return new OnnxStackConfig();
            }
        }
    }
}
