using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using System;

namespace OnnxStack.StableDiffusion
{
    public static class Extensions
    {
        /// <summary>
        /// Helper to get the scaled width for the latent dimension
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns></returns>
        /// <exception cref="System.ArgumentOutOfRangeException">Width must be divisible by 64</exception>
        public static int GetScaledWidth(this SchedulerOptions options)
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
        public static int GetScaledHeight(this SchedulerOptions options)
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
        public static int[] GetScaledDimension(this SchedulerOptions options, int batch = 1, int channels = 4)
        {
            return new[] { batch, channels, options.GetScaledHeight(), options.GetScaledWidth() };
        }



        /// <summary>
        /// Gets The starting inference step based on Strength value
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns>The starting inference step</returns>
        public static int GetStrengthScaledStartingStep(this SchedulerOptions options)
        {
            var inittimestep = Math.Min((int)(options.InferenceSteps * options.Strength), options.InferenceSteps);
            return Math.Min(Math.Max(options.InferenceSteps - inittimestep, 0), options.InferenceSteps - 1);
        }


        /// <summary>
        /// Sends a progress message.
        /// </summary>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="message">The message.</param>
        public static void Notify(this IProgress<DiffusionProgress> progressCallback, string message)
        {
            progressCallback?.Report(new DiffusionProgress(message));
        }
    }
}
