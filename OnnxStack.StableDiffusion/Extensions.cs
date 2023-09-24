using Microsoft.ML.OnnxRuntime;
using OnnxStack.StableDiffusion.Config;
using System;
using System.Linq;

namespace OnnxStack.StableDiffusion
{
    internal static class Extensions
    {
        /// <summary>
        /// Gets the first element and casts it to the specified type.
        /// </summary>
        /// <typeparam name="T">Desired return type</typeparam>
        /// <param name="collection">The collection.</param>
        /// <returns>Firts element in the collection cast as <see cref="T"/></returns>
        public static T FirstElementAs<T>(this IDisposableReadOnlyCollection<DisposableNamedOnnxValue> collection)
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
        public static T LastElementAs<T>(this IDisposableReadOnlyCollection<DisposableNamedOnnxValue> collection)
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
        public static int GetScaledWidth(this StableDiffusionOptions options)
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
        public static int GetScaledHeight(this StableDiffusionOptions options)
        {
            if (options.Height % 64 > 0)
                throw new ArgumentOutOfRangeException(nameof(options.Height), $"{nameof(options.Height)} must be divisible by 64");

                return options.Height / 8;
        }
    }
}
