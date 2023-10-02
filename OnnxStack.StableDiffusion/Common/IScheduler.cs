using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IScheduler : IDisposable
    {
        /// <summary>
        /// Gets the initial noise sigma.
        /// </summary>
        float InitNoiseSigma { get; }

        /// <summary>
        /// Gets the timesteps.
        /// </summary>
        IReadOnlyList<int> Timesteps { get; }

        /// <summary>
        /// Scales the input.
        /// </summary>
        /// <param name="sample">The sample.</param>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        DenseTensor<float> ScaleInput(DenseTensor<float> sample, int timestep);

        /// <summary>
        /// Processes a inference step for the specified model output.
        /// </summary>
        /// <param name="modelOutput">The model output.</param>
        /// <param name="timestep">The timestep.</param>
        /// <param name="sample">The sample.</param>
        /// <param name="order">The order.</param>
        /// <returns></returns>
        DenseTensor<float> Step(DenseTensor<float> modelOutput, int timestep, DenseTensor<float> sample, int order = 4);

        /// <summary>
        /// Adds noise to the sample.
        /// </summary>
        /// <param name="originalSamples">The original samples.</param>
        /// <param name="noise">The noise.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <returns></returns>
        DenseTensor<float> AddNoise(DenseTensor<float> originalSamples, DenseTensor<float> noise, IReadOnlyList<int> timesteps);

        /// <summary>
        /// Creates a random sample with the specified dimesions.
        /// </summary>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        DenseTensor<float> CreateRandomSample(ReadOnlySpan<int> dimensions, float initialNoiseSigma = 1.0f);
    }
}