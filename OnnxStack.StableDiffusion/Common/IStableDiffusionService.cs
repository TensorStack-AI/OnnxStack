using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.StableDiffusion.Config;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IStableDiffusionService
    {
        /// <summary>
        /// Generates the StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The Scheduler options.</param>
        /// <param name="progressCallback">The callback used to provide progess of the current InferenceSteps.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>The diffusion result as <see cref="DenseTensor<float>"/></returns>
        Task<DenseTensor<float>> GenerateAsync(PromptOptions prompt, SchedulerOptions options, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default);

        /// <summary>
        /// Generates the StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The Scheduler options.</param>
        /// <param name="progressCallback">The callback used to provide progess of the current InferenceSteps.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>The diffusion result as <see cref="SixLabors.ImageSharp.Image<Rgb24>"/></returns>
        Task<Image<Rgb24>> GenerateAsImageAsync(PromptOptions prompt, SchedulerOptions options, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default);

        /// <summary>
        /// Generates the StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The Scheduler options.</param>
        /// <param name="progressCallback">The callback used to provide progess of the current InferenceSteps.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>The diffusion result as <see cref="byte[]"/></returns>
        Task<byte[]> GenerateAsBytesAsync(PromptOptions prompt, SchedulerOptions options, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default);

        /// <summary>
        /// Generates the StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The Scheduler options.</param>
        /// <param name="progressCallback">The callback used to provide progess of the current InferenceSteps.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>The diffusion result as <see cref="System.IO.Stream"/></returns>
        Task<Stream> GenerateAsStreamAsync(PromptOptions prompt, SchedulerOptions options, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default);
    }
}