using Microsoft.Extensions.Logging;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace OnnxStack.Core
{
    public static class LogExtensions
    {

        public static void Log(this ILogger logger, string message, [CallerMemberName] string caller = default)
        {
            LogInternal(logger, LogLevel.Information, message, caller);
        }


        public static void Log(this ILogger logger, LogLevel logLevel, string message, [CallerMemberName] string caller = default)
        {
            LogInternal(logger, logLevel, message, caller);
        }


        public static long LogBegin(this ILogger logger, string message = default, [CallerMemberName] string caller = default)
        {
            return LogBeginInternal(logger, LogLevel.Information, message, caller);
        }


        public static long LogBegin(this ILogger logger, LogLevel logLevel, string message = default, [CallerMemberName] string caller = default)
        {

            return LogBeginInternal(logger, logLevel, message, caller);
        }


        public static void LogEnd(this ILogger logger, string message, long? timestamp, [CallerMemberName] string caller = default)
        {
            LogEndInternal(logger, LogLevel.Information, message, timestamp, caller);
        }


        public static void LogEnd(this ILogger logger, LogLevel logLevel, string message, long? timestamp, [CallerMemberName] string caller = default)
        {
            LogEndInternal(logger, logLevel, message, timestamp, caller);
        }


        private static long LogBeginInternal(ILogger logger, LogLevel logLevel, string message, string caller)
        {
            if (!string.IsNullOrEmpty(message))
                LogInternal(logger, logLevel, message, caller);

            return Stopwatch.GetTimestamp();
        }


        private static void LogEndInternal(ILogger logger, LogLevel logLevel, string message, long? timestamp, string caller)
        {
            var elapsed = Stopwatch.GetElapsedTime(timestamp ?? 0);
            var timeString = elapsed.TotalSeconds >= 1
                ? $"{message}, Elapsed: {elapsed.TotalSeconds:F4}sec"
                : $"{message}, Elapsed: {elapsed.TotalMilliseconds:F0}ms";
            LogInternal(logger, logLevel, timeString, caller);
        }

        private static void LogInternal(ILogger logger, LogLevel logLevel, string message, string caller)
        {
            logger?.Log(logLevel, string.IsNullOrEmpty(caller) ? message : $"[{caller}] - {message}", args: default);
        }
    }
}
