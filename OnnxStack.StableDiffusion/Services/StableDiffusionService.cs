using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
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
        private readonly IOnnxModelService _onnxModelService;
        private readonly StableDiffusionConfig _configuration;
        private readonly ConcurrentDictionary<DiffuserPipelineType, IPipeline> _pipelines;

        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionService"/> class.
        /// </summary>
        /// <param name="schedulerService">The scheduler service.</param>
        public StableDiffusionService(StableDiffusionConfig configuration, IOnnxModelService onnxModelService, IEnumerable<IPipeline> pipelines)
        {
            _configuration = configuration;
            _onnxModelService = onnxModelService;
            _pipelines = pipelines.ToConcurrentDictionary(k => k.PipelineType, k => k);
        }


        /// <summary>
        /// Gets the models.
        /// </summary>
        public List<ModelOptions> Models => _configuration.OnnxModelSets;


        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <returns></returns>
        public async Task<bool> LoadModel(IModelOptions modelOptions)
        {
            var model = await _onnxModelService.LoadModel(modelOptions);
            return model is not null;
        }


        /// <summary>
        /// Unloads the model.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <returns></returns>
        public async Task<bool> UnloadModel(IModelOptions modelOptions)
        {
            return await _onnxModelService.UnloadModel(modelOptions);
        }


        /// <summary>
        /// Is the model loaded.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <returns></returns>
        public bool IsModelLoaded(IModelOptions modelOptions)
        {
            return _onnxModelService.IsModelLoaded(modelOptions);
        }

        /// <summary>
        /// Generates the StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The Scheduler options.</param>
        /// <param name="progressCallback">The callback used to provide progess of the current InferenceSteps.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>The diffusion result as <see cref="DenseTensor<float>"/></returns>
        public async Task<DenseTensor<float>> GenerateAsync(IModelOptions model, PromptOptions prompt, SchedulerOptions options, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default)
        {
            return await DiffuseAsync(model, prompt, options, progressCallback, cancellationToken).ConfigureAwait(false);
        }


        /// <summary>
        /// Generates the StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The Scheduler options.</param>
        /// <param name="progressCallback">The callback used to provide progess of the current InferenceSteps.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>The diffusion result as <see cref="SixLabors.ImageSharp.Image<Rgba32>"/></returns>
        public async Task<Image<Rgba32>> GenerateAsImageAsync(IModelOptions model, PromptOptions prompt, SchedulerOptions options, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default)
        {
            return await GenerateAsync(model, prompt, options, progressCallback, cancellationToken)
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
        public async Task<byte[]> GenerateAsBytesAsync(IModelOptions model, PromptOptions prompt, SchedulerOptions options, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default)
        {
            return await GenerateAsync(model, prompt, options, progressCallback, cancellationToken)
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
        public async Task<Stream> GenerateAsStreamAsync(IModelOptions model, PromptOptions prompt, SchedulerOptions options, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default)
        {
            return await GenerateAsync(model, prompt, options, progressCallback, cancellationToken)
                .ContinueWith(t => t.Result.ToImageStream(), cancellationToken)
                .ConfigureAwait(false);
        }


        /// <summary>
        /// Generates a batch of StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="batchOptions">The batch options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public IAsyncEnumerable<BatchResult> GenerateBatchAsync(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<int, int, int, int> progressCallback = null, CancellationToken cancellationToken = default)
        {
            return DiffuseBatchAsync(modelOptions, promptOptions, schedulerOptions, batchOptions, progressCallback, cancellationToken);
        }


        /// <summary>
        /// Generates a batch of StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="batchOptions">The batch options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async IAsyncEnumerable<Image<Rgba32>> GenerateBatchAsImageAsync(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<int, int, int, int> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await foreach (var result in GenerateBatchAsync(modelOptions, promptOptions, schedulerOptions, batchOptions, progressCallback, cancellationToken))
                yield return result.ImageResult.ToImage();
        }


        /// <summary>
        /// Generates a batch of StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="batchOptions">The batch options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async IAsyncEnumerable<byte[]> GenerateBatchAsBytesAsync(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<int, int, int, int> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await foreach (var result in GenerateBatchAsync(modelOptions, promptOptions, schedulerOptions, batchOptions, progressCallback, cancellationToken))
                yield return result.ImageResult.ToImageBytes();
        }


        /// <summary>
        /// Generates a batch of StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="batchOptions">The batch options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async IAsyncEnumerable<Stream> GenerateBatchAsStreamAsync(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<int, int, int, int> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await foreach (var result in GenerateBatchAsync(modelOptions, promptOptions, schedulerOptions, batchOptions, progressCallback, cancellationToken))
                yield return result.ImageResult.ToImageStream();
        }


        private async Task<DenseTensor<float>> DiffuseAsync(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, Action<int, int> progress = null, CancellationToken cancellationToken = default)
        {
            if (!_pipelines.TryGetValue(modelOptions.PipelineType, out var pipeline))
                throw new Exception("Pipeline not found or is unsupported");

            var diffuser = pipeline.GetDiffuser(promptOptions.DiffuserType);
            if (diffuser is null)
                throw new Exception("Diffuser not found or is unsupported");

            return await diffuser.DiffuseAsync(modelOptions, promptOptions, schedulerOptions, progress, cancellationToken);
        }


        private IAsyncEnumerable<BatchResult> DiffuseBatchAsync(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<int, int, int, int> progress = null, CancellationToken cancellationToken = default)
        {
            if (!_pipelines.TryGetValue(modelOptions.PipelineType, out var pipeline))
                throw new Exception("Pipeline not found or is unsupported");

            var diffuser = pipeline.GetDiffuser(promptOptions.DiffuserType);
            if (diffuser is null)
                throw new Exception("Diffuser not found or is unsupported");

            return diffuser.DiffuseBatchAsync(modelOptions, promptOptions, schedulerOptions, batchOptions, progress, cancellationToken);
        }
    }
}
