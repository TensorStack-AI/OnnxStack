using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Configuration;
using System.Data;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using OnnxStack.Core;
using OnnxStack.UI.Services;
using OnnxStack.UI.Dialogs;
using System.Diagnostics;
using System.Windows.Controls;
using System.Windows.Threading;

namespace OnnxStack.UI
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        private static IHost _applicationHost;
        private static ILogger<App> _logger;

        public App()
        {
            var builder = Host.CreateApplicationBuilder();
            builder.Logging.ClearProviders();
            builder.Services.AddLogging((loggingBuilder) => loggingBuilder.SetMinimumLevel(LogLevel.Trace).AddWindowLogger());

            // Add OnnxStackStableDiffusion
            builder.Services.AddOnnxStackStableDiffusion();

            // Add Windows
            builder.Services.AddSingleton<MainWindow>();
            builder.Services.AddTransient<MessageDialog>();
            builder.Services.AddTransient<TextInputDialog>();
            builder.Services.AddTransient<CropImageDialog>();
            builder.Services.AddSingleton<IDialogService, DialogService>();

            // Build App
            _applicationHost = builder.Build();
        }


        public static T GetService<T>() => _applicationHost.Services.GetService<T>();

        public static void UIInvoke(Action action, DispatcherPriority priority = DispatcherPriority.Render) => Current.Dispatcher.BeginInvoke(priority, action);


        /// <summary>
        /// Raises the <see cref="E:Startup" /> event.
        /// </summary>
        /// <param name="e">The <see cref="StartupEventArgs"/> instance containing the event data.</param>
        protected override async void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);
            await _applicationHost.StartAsync();
            GetService<MainWindow>().Show();
        }


        /// <summary>
        /// Raises the <see cref="E:Exit" /> event.
        /// </summary>
        /// <param name="e">The <see cref="ExitEventArgs"/> instance containing the event data.</param>
        protected override async void OnExit(ExitEventArgs e)
        {
            await _applicationHost.StopAsync();
            base.OnExit(e);
        }
    }
}
