using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Logging;
using System;

namespace OnnxStack.UI
{

    /// <summary>
    /// Simple logger to capture output to the main windows OutputLog
    /// </summary>
    /// <seealso cref="Microsoft.Extensions.Logging.ILogger" />
    public sealed class WindowLogger : ILogger
    {
        /// <summary>
        /// Begins a logical operation scope.
        /// </summary>
        /// <typeparam name="TState">The type of the state to begin scope for.</typeparam>
        /// <param name="state">The identifier for the scope.</param>
        /// <returns>
        /// A disposable object that ends the logical operation scope on dispose.
        /// </returns>
        public IDisposable BeginScope<TState>(TState state) where TState : notnull => default!;


        /// <summary>
        /// Checks if the given <paramref name="logLevel" /> is enabled.
        /// </summary>
        /// <param name="logLevel">level to be checked.</param>
        /// <returns>
        ///   <see langword="true" /> if enabled; <see langword="false" /> otherwise.
        /// </returns>
        public bool IsEnabled(LogLevel logLevel) => true;


        /// <summary>
        /// Writes a log entry.
        /// </summary>
        /// <typeparam name="TState">The type of the object to be written.</typeparam>
        /// <param name="logLevel">Entry will be written on this level.</param>
        /// <param name="eventId">Id of the event.</param>
        /// <param name="state">The entry to be written. Can be also an object.</param>
        /// <param name="exception">The exception related to this entry.</param>
        /// <param name="formatter">Function to create a <see cref="T:System.String" /> message of the <paramref name="state" /> and <paramref name="exception" />.</param>
        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception exception, Func<TState, Exception, string> formatter)
        {
            if (!IsEnabled(logLevel))
                return;

            Utils.LogToWindow($"[{DateTime.Now}] [{logLevel}] - {formatter(state, exception)}\n");
        }



    }


    public static class WindowLoggerLoggerExtensions
    {
        /// <summary>
        /// Adds the window logger.
        /// </summary>
        /// <param name="builder">The builder.</param>
        /// <returns></returns>
        public static ILoggingBuilder AddWindowLogger(this ILoggingBuilder builder)
        {
            builder.Services.TryAddEnumerable(ServiceDescriptor.Singleton<ILoggerProvider, WindowLoggerProvider>());
            return builder;
        }
    }

    public sealed class WindowLoggerProvider : ILoggerProvider
    {

        /// <summary>
        /// Creates a new <see cref="T:Microsoft.Extensions.Logging.ILogger" /> instance.
        /// </summary>
        /// <param name="categoryName">The category name for messages produced by the logger.</param>
        /// <returns>
        /// The instance of <see cref="T:Microsoft.Extensions.Logging.ILogger" /> that was created.
        /// </returns>
        public ILogger CreateLogger(string categoryName) => new WindowLogger();

        public void Dispose()
        {
        }
    }
}
