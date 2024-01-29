using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace OnnxStack.Console
{
    internal class AppService : IHostedService
    {
        private readonly IReadOnlyList<IExampleRunner> _exampleRunners;

        public AppService(IServiceProvider serviceProvider, IEnumerable<Type> exampleRunnerTypes)
        {
            _exampleRunners = exampleRunnerTypes
                .Select(serviceProvider.GetService)
                .Cast<IExampleRunner>()
                .OrderBy(x => x.Index)
                .ToList();
        }

        public async Task StartAsync(CancellationToken cancellationToken)
        {
            var index = 1;
            OutputHelpers.WriteConsole("Please enter an example number below and press enter:\n", ConsoleColor.Cyan);
            foreach (var exampleRunner in _exampleRunners)
            {
                OutputHelpers.WriteConsole($"{index++}. {exampleRunner.Name} - {exampleRunner.Description}", ConsoleColor.Gray);
            }

            while (true)
            {

                OutputHelpers.WriteConsole("\nExample: ", ConsoleColor.Green, false);
                var selection = OutputHelpers.ReadConsole(ConsoleColor.Gray);
                if (!int.TryParse(selection, out index) || index > _exampleRunners.Count)
                {
                    OutputHelpers.WriteConsole($"{selection} is an invalid option number", ConsoleColor.Red);
                    continue;
                }

                var selectedRunner = _exampleRunners[index - 1];
                OutputHelpers.WriteConsole($"Starting example '{selectedRunner.Name}'\n", ConsoleColor.Yellow);
                await selectedRunner.RunAsync();
            }
        }

        public Task StopAsync(CancellationToken cancellationToken)
        {
            return Task.CompletedTask;
        }
    }
}