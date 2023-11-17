using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace OnnxStack.Core
{
    public static class Extensions
    {
        public static SessionOptions GetSessionOptions(this OnnxModelSessionConfig configuration)
        {
            var sessionOptions = new SessionOptions
            {
                ExecutionMode = configuration.ExecutionMode.Value,
                InterOpNumThreads = configuration.InterOpNumThreads.Value,
                IntraOpNumThreads = configuration.IntraOpNumThreads.Value
            };
            switch (configuration.ExecutionProvider)
            {
                case ExecutionProvider.DirectML:
                    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    sessionOptions.EnableMemoryPattern = false;
                    sessionOptions.AppendExecutionProvider_DML(configuration.DeviceId.Value);
                    sessionOptions.AppendExecutionProvider_CPU();
                    return sessionOptions;
                case ExecutionProvider.Cpu:
                    sessionOptions.AppendExecutionProvider_CPU();
                    return sessionOptions;
                default:
                case ExecutionProvider.Cuda:
                    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    sessionOptions.AppendExecutionProvider_CUDA(configuration.DeviceId.Value);
                    sessionOptions.AppendExecutionProvider_CPU();
                    return sessionOptions;
                case ExecutionProvider.CoreML:
                    sessionOptions.AppendExecutionProvider_CoreML(
                        CoreMLFlags.COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE
                    );
                    return sessionOptions;
            }
        }

        /// <summary>
        /// Applies the configuration overrides.
        /// </summary>
        public static void ApplyConfigurationOverrides(this IOnnxModelSetConfig innxModelSetConfig)
        {
            if (innxModelSetConfig.ModelConfigurations.IsNullOrEmpty())
                return;

            foreach (var modelConfig in innxModelSetConfig.ModelConfigurations)
            {
                modelConfig.DeviceId ??= innxModelSetConfig.DeviceId;
                modelConfig.ExecutionMode ??= innxModelSetConfig.ExecutionMode;
                modelConfig.InterOpNumThreads ??= innxModelSetConfig.InterOpNumThreads;
                modelConfig.IntraOpNumThreads ??= innxModelSetConfig.IntraOpNumThreads;
                modelConfig.ExecutionProvider ??= innxModelSetConfig.ExecutionProvider;
            }
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
        ///   Batches the source sequence into sized buckets.
        /// </summary>
        /// <typeparam name="TSource">Type of elements in <paramref name="source" /> sequence.</typeparam>
        /// <param name="source">The source sequence.</param>
        /// <param name="size">Size of buckets.</param>
        /// <returns>A sequence of equally sized buckets containing elements of the source collection.</returns>
        /// <remarks>
        ///   This operator uses deferred execution and streams its results (buckets and bucket content).
        /// </remarks>
        public static IEnumerable<IEnumerable<TSource>> Batch<TSource>(this IEnumerable<TSource> source, int size)
        {
            return Batch(source, size, x => x);
        }

        /// <summary>
        ///   Batches the source sequence into sized buckets and applies a projection to each bucket.
        /// </summary>
        /// <typeparam name="TSource">Type of elements in <paramref name="source" /> sequence.</typeparam>
        /// <typeparam name="TResult">Type of result returned by <paramref name="resultSelector" />.</typeparam>
        /// <param name="source">The source sequence.</param>
        /// <param name="size">Size of buckets.</param>
        /// <param name="resultSelector">The projection to apply to each bucket.</param>
        /// <returns>A sequence of projections on equally sized buckets containing elements of the source collection.</returns>
        /// <remarks>
        ///   This operator uses deferred execution and streams its results (buckets and bucket content).
        /// </remarks>
        public static IEnumerable<TResult> Batch<TSource, TResult>(this IEnumerable<TSource> source, int size, Func<IEnumerable<TSource>, TResult> resultSelector)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));
            if (size <= 0)
                throw new ArgumentOutOfRangeException(nameof(size));
            if (resultSelector == null)
                throw new ArgumentNullException(nameof(resultSelector));
            return BatchImpl(source, size, resultSelector);
        }


        private static IEnumerable<TResult> BatchImpl<TSource, TResult>(this IEnumerable<TSource> source, int size, Func<IEnumerable<TSource>, TResult> resultSelector)
        {
            TSource[] bucket = null;
            var count = 0;
            foreach (var item in source)
            {
                if (bucket == null)
                    bucket = new TSource[size];

                bucket[count++] = item;

                // The bucket is fully buffered before it's yielded
                if (count != size)
                    continue;

                // Select is necessary so bucket contents are streamed too
                yield return resultSelector(bucket.Select(x => x));
                bucket = null;
                count = 0;
            }

            // Return the last bucket with all remaining elements
            if (bucket != null && count > 0)
                yield return resultSelector(bucket.Take(count));
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
        /// Converts to long.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <returns></returns>
        public static long[] ToLong(this int[] array)
        {
            return Array.ConvertAll(array, Convert.ToInt64);
        }


        /// <summary>
        /// Creates and OrtValue form the DenseTensor and NodeMetaData provided
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="metadata">The metadata.</param>
        /// <returns></returns>
        public static OrtValue ToOrtValue(this DenseTensor<float> tensor, OnnxNamedMetadata metadata)
        {
            var dimensions = tensor.Dimensions.ToLong();
            return metadata.Value.ElementDataType switch
            {
                TensorElementType.Float16 => OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Buffer.ToFloat16(), dimensions),
                TensorElementType.BFloat16 => OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Buffer.ToBFloat16(), dimensions),
                _ => OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Buffer, dimensions)
            };
        }


        /// <summary>
        /// Creates and allocates the output tensors buffer.
        /// </summary>
        /// <param name="metadata">The metadata.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        public static OrtValue CreateOutputBuffer(this OnnxNamedMetadata metadata, ReadOnlySpan<int> dimensions)
        {
            return OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, metadata.Value.ElementDataType, dimensions.ToLong());
        }


        /// <summary>
        /// Converts to DenseTensor<float>.
        /// </summary>
        /// <param name="ortValue">The ort value.</param>
        /// <returns></returns>
        public static DenseTensor<float> ToDenseTensor(this OrtValue ortValue)
        {
            var typeInfo = ortValue.GetTensorTypeAndShape();
            var dimensions = typeInfo.Shape.ToInt();
            return typeInfo.ElementDataType switch
            {
                TensorElementType.Float16 => new DenseTensor<float>(ortValue.GetTensorDataAsSpan<Float16>().ToFloat(), dimensions),
                TensorElementType.BFloat16 => new DenseTensor<float>(ortValue.GetTensorDataAsSpan<BFloat16>().ToFloat(), dimensions),
                _ => new DenseTensor<float>(ortValue.GetTensorDataAsSpan<float>().ToArray(), dimensions)
            };
        }


        /// <summary>
        /// Converts to array.
        /// </summary>
        /// <param name="ortValue">The ort value.</param>
        /// <returns></returns>
        public static float[] ToArray(this OrtValue ortValue)
        {
            var typeInfo = ortValue.GetTensorTypeAndShape();
            var dimensions = typeInfo.Shape.ToInt();
            return typeInfo.ElementDataType switch
            {
                TensorElementType.Float16 => ortValue.GetTensorDataAsSpan<Float16>().ToFloat().ToArray(),
                TensorElementType.BFloat16 => ortValue.GetTensorDataAsSpan<BFloat16>().ToFloat().ToArray(),
                _ => ortValue.GetTensorDataAsSpan<float>().ToArray()
            };
        }


        /// <summary>
        /// Converts to float16.
        /// </summary>
        /// <param name="inputMemory">The input memory.</param>
        /// <returns></returns>
        internal static Memory<Float16> ToFloat16(this Memory<float> inputMemory)
        {
            var elementCount = inputMemory.Length;
            var floatArray = new Float16[inputMemory.Length];
            for (int i = 0; i < elementCount; i++)
                floatArray[i] = (Float16)inputMemory.Span[i];

            return floatArray.AsMemory();
        }


        /// <summary>
        /// Converts to BFloat16.
        /// </summary>
        /// <param name="inputMemory">The input memory.</param>
        /// <returns></returns>
        internal static Memory<BFloat16> ToBFloat16(this Memory<float> inputMemory)
        {
            var elementCount = inputMemory.Length;
            var floatArray = new BFloat16[inputMemory.Length];
            for (int i = 0; i < elementCount; i++)
                floatArray[i] = (BFloat16)inputMemory.Span[i];

            return floatArray.AsMemory();
        }


        /// <summary>
        /// Converts to float.
        /// </summary>
        /// <param name="inputMemory">The input memory.</param>
        /// <returns></returns>
        internal static Memory<float> ToFloat(this ReadOnlySpan<Float16> inputMemory)
        {
            var elementCount = inputMemory.Length;
            var floatArray = new float[elementCount];
            for (int i = 0; i < elementCount; i++)
                floatArray[i] = (float)inputMemory[i];

            return floatArray.AsMemory();
        }


        /// <summary>
        /// Converts to float.
        /// </summary>
        /// <param name="inputMemory">The input memory.</param>
        /// <returns></returns>
        internal static Memory<float> ToFloat(this ReadOnlySpan<BFloat16> inputMemory)
        {
            var elementCount = inputMemory.Length;
            var floatArray = new float[elementCount];
            for (int i = 0; i < elementCount; i++)
                floatArray[i] = (float)inputMemory[i];

            return floatArray.AsMemory();
        }
    }
}
