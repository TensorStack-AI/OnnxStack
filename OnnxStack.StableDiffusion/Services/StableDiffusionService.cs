using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Helpers;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Services
{
    /// <summary>
    /// Service for generating images using text and image based prompts
    /// </summary>
    /// <seealso cref="OnnxStack.StableDiffusion.Common.IStableDiffusionService" />
    public sealed class StableDiffusionService : IStableDiffusionService
    {
        private readonly ISchedulerService _schedulerService;

        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionService"/> class.
        /// </summary>
        /// <param name="schedulerService">The scheduler service.</param>
        public StableDiffusionService(ISchedulerService schedulerService)
        {
            _schedulerService = schedulerService;
        }


        /// <summary>
        /// Generates the StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The Scheduler options.</param>
        /// <param name="progressCallback">The callback used to provide progess of the current InferenceSteps.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>The diffusion result as <see cref="DenseTensor<float>"/></returns>
        public async Task<DenseTensor<float>> GenerateAsync(PromptOptions prompt, SchedulerOptions options, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default)
        {
            return await _schedulerService.RunAsync(prompt, options, progressCallback, cancellationToken).ConfigureAwait(false);
        }


        /// <summary>
        /// Generates the StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The Scheduler options.</param>
        /// <param name="progressCallback">The callback used to provide progess of the current InferenceSteps.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>The diffusion result as <see cref="SixLabors.ImageSharp.Image<Rgb24>"/></returns>
        public async Task<Image<Rgb24>> GenerateAsImageAsync(PromptOptions prompt, SchedulerOptions options, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default)
        {
            return await GenerateAsync(prompt, options, progressCallback, cancellationToken)
                .ContinueWith(t => t.Result.ToImage(), cancellationToken)
                .ConfigureAwait(false);
        }


        /// <summary>
        /// Generates the StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The Scheduler options.</param>
        /// <param name="progressCallback">The callback used to provide progess of the current InferenceSteps.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>The diffusion result as <see cref="byte[]"/></returns>
        public async Task<byte[]> GenerateAsBytesAsync(PromptOptions prompt, SchedulerOptions options, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default)
        {
            return await GenerateAsync(prompt, options, progressCallback, cancellationToken)
                .ContinueWith(t => t.Result.ToImageBytes(), cancellationToken)
                .ConfigureAwait(false);
        }


        /// <summary>
        /// Generates the StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The Scheduler options.</param>
        /// <param name="progressCallback">The callback used to provide progess of the current InferenceSteps.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>The diffusion result as <see cref="System.IO.Stream"/></returns>
        public async Task<Stream> GenerateAsStreamAsync(PromptOptions prompt, SchedulerOptions options, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default)
        {
            return await GenerateAsync(prompt, options, progressCallback, cancellationToken)
                .ContinueWith(t => t.Result.ToImageStream(), cancellationToken)
                .ConfigureAwait(false);
        }
    }
}
