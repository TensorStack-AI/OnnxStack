using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using OnnxStack.Core;
using System.Reflection;

namespace OnnxStack.Console
{
    internal class Program
    {
        static async Task Main(string[] _)
        {
            var builder = Host.CreateApplicationBuilder();
            builder.Logging.ClearProviders();
            builder.Services.AddLogging((loggingBuilder) => loggingBuilder.SetMinimumLevel(LogLevel.Error));

            // Add OnnxStack
            builder.Services.AddOnnxStack();
            builder.Services.AddOnnxStackStableDiffusion();

            // Add AppService
            builder.Services.AddHostedService<AppService>();

            // Add Runners
            var exampleRunners = Assembly.GetExecutingAssembly()
                .GetTypes()
                .Where(type => typeof(IExampleRunner).IsAssignableFrom(type) && !type.IsInterface)
                .ToList();
            builder.Services.AddSingleton(exampleRunners.AsEnumerable());
            foreach (var exampleRunner in exampleRunners)
            {
                builder.Services.AddSingleton(exampleRunner);
            }

            // Start
            await builder.Build().RunAsync();
        }

    }
}