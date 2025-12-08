using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace OnnxStack.Core
{
    public static class Extensions
    {

        public static T ApplyDefaults<T>(this T config, IOnnxModelSetConfig defaults) where T : OnnxModelConfig
        {
            return config with
            {
                ExecutionProvider = config.ExecutionProvider ?? defaults.ExecutionProvider
            };
        }


        /// <summary>
        /// Determines whether the the source sequence is null or empty
        /// </summary>
        /// <typeparam name="TSource">Type of elements in <paramref name="source" /> sequence.</typeparam>
        /// <param name="source">The source sequence.</param>
        /// <returns>
        ///   <c>true</c> if the source sequence is null or empty; otherwise, <c>false</c>.
        /// </returns>
        public static bool IsNullOrEmpty<TSource>(this IEnumerable<TSource> source)
        {
            return source == null || !source.Any();
        }


        /// <summary>
        /// Get the index of the specified item
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="list">The list.</param>
        /// <param name="item">The item.</param>
        /// <returns></returns>
        public static int IndexOf<T>(this IReadOnlyList<T> list, T item) where T : IEquatable<T>
        {
            for (int i = 0; i < list.Count; i++)
            {
                if (list[i].Equals(item))
                    return i;
            }
            return -1;
        }


        /// <summary>
        /// Get the index of the specified item
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="list">The list.</param>
        /// <param name="itemSelector">The item selector.</param>
        /// <returns>System.Int32.</returns>
        public static int IndexOf<T>(this IReadOnlyList<T> list, Func<T, bool> itemSelector) where T : IEquatable<T>
        {
            var item = list.FirstOrDefault(itemSelector);
            if (item == null)
                return -1;

            return IndexOf(list, item);
        }


        /// <summary>
        /// Converts to source IEnumerable to a ConcurrentDictionary.
        /// </summary>
        /// <param name="source">The source.</param>
        /// <param name="keySelector">The key selector.</param>
        /// <param name="elementSelector">The element selector.</param>
        /// <returns></returns>
        public static ConcurrentDictionary<T, U> ToConcurrentDictionary<S, T, U>(this IEnumerable<S> source, Func<S, T> keySelector, Func<S, U> elementSelector) where T : notnull
        {
            return new ConcurrentDictionary<T, U>(source.ToDictionary(keySelector, elementSelector));
        }


        /// <summary>
        /// Gets the full prod of a dimension
        /// </summary>
        /// <param name="array">The dimension array.</param>
        /// <returns></returns>
        public static T GetBufferLength<T>(this T[] array) where T : INumber<T>
        {
            T result = T.One;
            foreach (T element in array)
            {
                result *= element;
            }
            return result;
        }


        /// <summary>
        /// Gets the full prod of a dimension
        /// </summary>
        /// <param name="array">The dimension array.</param>
        /// <returns></returns>
        public static T GetBufferLength<T>(this ReadOnlySpan<T> array) where T : INumber<T>
        {
            T result = T.One;
            foreach (T element in array)
            {
                result *= element;
            }
            return result;
        }


        /// <summary>
        /// Converts to long.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <returns></returns>
        public static long[] ToLong(this ReadOnlySpan<int> array)
        {
            return Array.ConvertAll(array.ToArray(), Convert.ToInt64);
        }


        /// <summary>
        /// Converts the string representation of a number to an integer.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <returns></returns>
        public static int[] ToInt(this long[] array)
        {
            return Array.ConvertAll(array, Convert.ToInt32);
        }


        /// <summary>
        /// Converts to intsafe.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <returns></returns>
        /// <exception cref="OverflowException">$"Value {value} at index {i} is outside the range of an int.</exception>
        public static int[] ToIntSafe(this long[] array)
        {
            int[] result = new int[array.Length];

            for (int i = 0; i < array.Length; i++)
            {
                long value = array[i];

                if (value < int.MinValue || value > int.MaxValue)
                    value = 0;

                result[i] = (int)value;
            }

            return result;
        }


        /// <summary>
        /// Converts to long.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <returns></returns>
        public static long[] ToLong(this int[] array)
        {
            return Array.ConvertAll(array, Convert.ToInt64);
        }


        /// <summary>
        /// Normalize the data using Min-Max scaling to ensure all values are in the range [0, 1].
        /// </summary>
        /// <param name="values">The values.</param>
        public static Span<float> NormalizeZeroToOne(this Span<float> values)
        {
            float min = float.PositiveInfinity, max = float.NegativeInfinity;
            foreach (var val in values)
            {
                if (min > val) min = val;
                if (max < val) max = val;
            }

            var range = max - min;
            for (var i = 0; i < values.Length; i++)
            {
                values[i] = (values[i] - min) / range;
            }
            return values;
        }


        public static Span<float> NormalizeOneToOne(this Span<float> values)
        {
            float max = values[0];
            foreach (var val in values)
            {
                if (max < val) max = val;
            }

            for (var i = 0; i < values.Length; i++)
            {
                values[i] = (values[i] * 2) - 1;
            }
            return values;
        }


        public static void RemoveRange<TSource>(this List<TSource> source, IEnumerable<TSource> toRemove)
        {
            if (toRemove.IsNullOrEmpty())
                return;

            foreach (var item in toRemove)
                source.Remove(item);
        }


        public static void CancelSession(this SessionOptions sessionOptions)
        {
            sessionOptions.SetLoadCancellationFlag(true);
        }


        public static void CancelSession(this RunOptions runOptions)
        {
            try
            {
                if (runOptions.IsClosed)
                    return;

                if (runOptions.IsInvalid)
                    return;

                if (runOptions.Terminate == true)
                    return;

                runOptions.Terminate = true;
            }
            catch (Exception)
            {
                throw new OperationCanceledException();
            }
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ZeroIfNan(this float value)
        {
            return float.IsNaN(value) ? 0f : value;
        }
    }
}
