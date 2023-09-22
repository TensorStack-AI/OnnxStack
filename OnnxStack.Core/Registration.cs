using OnnxStack.Common.Config;
using OnnxStack.Core.Config;
using Microsoft.Extensions.DependencyInjection;
using System;

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
            serviceCollection.AddOnnxStack<Guid>();
        }


        /// <summary>
        /// Register OnnxStack services
        /// </summary>
        /// <typeparam name="T">The type used for session identification</typeparam>
        /// <param name="serviceCollection">The service collection.</param>
        public static void AddOnnxStack<T>(this IServiceCollection serviceCollection) 
            where T : IEquatable<T>, IComparable<T>
        {
            serviceCollection.AddSingleton(ConfigManager.LoadConfiguration());
           // serviceCollection.AddHostedService<ModelLoaderService<T>>();
          //  serviceCollection.AddSingleton<IModelService<T>, ModelService<T>>();
          //  serviceCollection.AddSingleton<IModelSessionService<T>, ModelSessionService<T>>();
          //  serviceCollection.AddSingleton<IModelSessionStateService<T>, ModelSessionStateService<T>>();
        }


        /// <summary>
        /// Register a custom IConfigSection section that is in the appsettings.json
        /// </summary>
        /// <typeparam name="T">The custom IConfigSection class type, NOTE: json section name MUST match class name</typeparam>
        /// <param name="serviceCollection">The service collection.</param>
        public static void AddOnnxCustomConfig<T>(this IServiceCollection serviceCollection) 
            where T : class, IConfigSection
        {
            serviceCollection.AddSingleton(ConfigManager.LoadConfiguration<T>());
        }
    }
}
