using Microsoft.ML.OnnxRuntime;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using System;
using System.Linq;

namespace OnnxStack.StableDiffusion
{
    public static class Extensions
    {
        /// <summary>
        /// Gets the first element and casts it to the specified type.
        /// </summary>
        /// <typeparam name="T">Desired return type</typeparam>
        /// <param name="collection">The collection.</param>
        /// <returns>Firts element in the collection cast as <see cref="T"/></returns>
        internal static T FirstElementAs<T>(this IDisposableReadOnlyCollection<DisposableNamedOnnxValue> collection)
        {
            if (collection is null || collection.Count == 0)
                return default;

            var element = collection.FirstOrDefault();
            if (element.Value is not T value)
                return default;

            return value;
        }


        /// <summary>
        /// Gets the last element and casts it to the specified type.
        /// </summary>
        /// <typeparam name="T">Desired return type</typeparam>
        /// <param name="collection">The collection.</param>
        /// <returns>Last element in the collection cast as <see cref="T"/></returns>
        internal static T LastElementAs<T>(this IDisposableReadOnlyCollection<DisposableNamedOnnxValue> collection)
        {
            if (collection is null || collection.Count == 0)
                return default;

            var element = collection.LastOrDefault();
            if (element.Value is not T value)
                return default;

            return value;
        }


        /// <summary>
        /// Helper to get the scaled width for the latent dimension
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns></returns>
        /// <exception cref="System.ArgumentOutOfRangeException">Width must be divisible by 64</exception>
        internal static int GetScaledWidth(this SchedulerOptions options)
        {
            if (options.Width % 64 > 0)
                throw new ArgumentOutOfRangeException(nameof(options.Width), $"{nameof(options.Width)} must be divisible by 64");

            return options.Width / 8;
        }


        /// <summary>
        /// Helper to get the scaled height for the latent dimension
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns></returns>
        /// <exception cref="System.ArgumentOutOfRangeException">Height must be divisible by 64</exception>
        internal static int GetScaledHeight(this SchedulerOptions options)
        {
            if (options.Height % 64 > 0)
                throw new ArgumentOutOfRangeException(nameof(options.Height), $"{nameof(options.Height)} must be divisible by 64");

            return options.Height / 8;
        }


        /// <summary>
        /// Gets a tensor dimension for the input image in the shape of [batch, channels, (Height / 8), (Width / 8)].
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="batch">The batch.</param>
        /// <param name="channels">The channels.</param>
        /// <returns>Tensor dimension of [batch, channels, (Height / 8), (Width / 8)]</returns>
        internal static int[] GetScaledDimension(this SchedulerOptions options, int batch = 1, int channels = 4)
        {
            return new[] { batch, channels, options.GetScaledHeight(), options.GetScaledWidth() };
        }


        /// <summary>
        /// Gets the pipeline schedulers.
        /// </summary>
        /// <param name="pipelineType">Type of the pipeline.</param>
        /// <returns></returns>
        public static SchedulerType[] GetSchedulerTypes(this DiffuserPipelineType pipelineType)
        {
            return pipelineType switch
            {
                DiffuserPipelineType.StableDiffusion => new[]
                {
                    SchedulerType.LMS,
                    SchedulerType.Euler,
                    SchedulerType.EulerAncestral,
                    SchedulerType.DDPM,
                    SchedulerType.DDIM,
                    SchedulerType.KDPM2
                },
                DiffuserPipelineType.LatentConsistency => new[]
                {
                    SchedulerType.LCM
                },
                _ => default
            };
        }
    }
}
