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
using System.Linq;
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
        private readonly IOnnxModelService _modelService;
        private readonly StableDiffusionConfig _configuration;
        private readonly ConcurrentDictionary<DiffuserPipelineType, IPipeline> _pipelines;

        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionService"/> class.
        /// </summary>
        /// <param name="schedulerService">The scheduler service.</param>
        public StableDiffusionService(StableDiffusionConfig configuration, IOnnxModelService onnxModelService, IEnumerable<IPipeline> pipelines)
        {
            _configuration = configuration;
            _modelService = onnxModelService;
            _modelService.AddModelSet(configuration.ModelSets);
            _pipelines = pipelines.ToConcurrentDictionary(k => k.PipelineType, k => k);
        }


        /// <summary>
        /// Gets the configuration.
        /// </summary>
        public StableDiffusionConfig Configuration => _configuration;


        /// <summary>
        /// Gets the model sets.
        /// </summary>
        public IReadOnlyList<StableDiffusionModelSet> ModelSets => _configuration.ModelSets;


        /// <summary>
        /// Adds the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        public Task<bool> AddModelAsync(StableDiffusionModelSet model)
        {
            return _modelService.AddModelSet(model);
        }


        /// <summary>
        /// Removes the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        public Task<bool> RemoveModelAsync(StableDiffusionModelSet model)
        {
            return _modelService.RemoveModelSet(model);
        }


        /// <summary>
        /// Updates the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        public Task<bool> UpdateModelAsync(StableDiffusionModelSet model)
        {
            return _modelService.UpdateModelSet(model);
        }


        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <param name="model">The model options.</param>
        /// <returns></returns>
        public async Task<bool> LoadModelAsync(StableDiffusionModelSet modelSet)
        {
            var model = await _modelService.LoadModelAsync(modelSet);
            return model is not null;
        }


        /// <summary>
        /// Unloads the model.
        /// </summary>
        /// <param name="modelSet">The model options.</param>
        /// <returns></returns>
        public async Task<bool> UnloadModelAsync(StableDiffusionModelSet modelSet)
        {
            return await _modelService.UnloadModelAsync(modelSet);
        }


        /// <summary>
        /// Is the model loaded.
        /// </summary>
        /// <param name="modelSet">The model options.</param>
        /// <returns></returns>
        public bool IsModelLoaded(StableDiffusionModelSet modelSet)
        {
            return _modelService.IsModelLoaded(modelSet);
        }

        /// <summary>
        /// Generates the StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The Scheduler options.</param>
        /// <param name="progressCallback">The callback used to provide progess of the current InferenceSteps.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>The diffusion result as <see cref="DenseTensor<float>"/></returns>
        public async Task<DenseTensor<float>> GenerateAsync(StableDiffusionModelSet model, PromptOptions prompt, SchedulerOptions options, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default)
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
        public async Task<Image<Rgba32>> GenerateAsImageAsync(StableDiffusionModelSet model, PromptOptions prompt, SchedulerOptions options, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default)
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
        public async Task<byte[]> GenerateAsBytesAsync(StableDiffusionModelSet model, PromptOptions prompt, SchedulerOptions options, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default)
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
        public async Task<Stream> GenerateAsStreamAsync(StableDiffusionModelSet model, PromptOptions prompt, SchedulerOptions options, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default)
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
        public IAsyncEnumerable<BatchResult> GenerateBatchAsync(StableDiffusionModelSet modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<int, int, int, int> progressCallback = null, CancellationToken cancellationToken = default)
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
        public async IAsyncEnumerable<Image<Rgba32>> GenerateBatchAsImageAsync(StableDiffusionModelSet modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<int, int, int, int> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
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
        public async IAsyncEnumerable<byte[]> GenerateBatchAsBytesAsync(StableDiffusionModelSet modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<int, int, int, int> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
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
        public async IAsyncEnumerable<Stream> GenerateBatchAsStreamAsync(StableDiffusionModelSet modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<int, int, int, int> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await foreach (var result in GenerateBatchAsync(modelOptions, promptOptions, schedulerOptions, batchOptions, progressCallback, cancellationToken))
                yield return result.ImageResult.ToImageStream();
        }


        private async Task<DenseTensor<float>> DiffuseAsync(StableDiffusionModelSet modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, Action<int, int> progress = null, CancellationToken cancellationToken = default)
        {
            if (!_pipelines.TryGetValue(modelOptions.PipelineType, out var pipeline))
                throw new Exception("Pipeline not found or is unsupported");

            var diffuser = pipeline.GetDiffuser(promptOptions.DiffuserType);
            if (diffuser is null)
                throw new Exception("Diffuser not found or is unsupported");

            var schedulerSupported = pipeline.PipelineType.GetSchedulerTypes().Contains(schedulerOptions.SchedulerType);
            if (!schedulerSupported)
                throw new Exception($"Scheduler '{schedulerOptions.SchedulerType}' is not compatible  with the `{pipeline.PipelineType}` pipeline.");

            return await diffuser.DiffuseAsync(modelOptions, promptOptions, schedulerOptions, progress, cancellationToken);
        }


        private IAsyncEnumerable<BatchResult> DiffuseBatchAsync(StableDiffusionModelSet modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<int, int, int, int> progress = null, CancellationToken cancellationToken = default)
        {
            if (!_pipelines.TryGetValue(modelOptions.PipelineType, out var pipeline))
                throw new Exception("Pipeline not found or is unsupported");

            var diffuser = pipeline.GetDiffuser(promptOptions.DiffuserType);
            if (diffuser is null)
                throw new Exception("Diffuser not found or is unsupported");

            var schedulerSupported = pipeline.PipelineType.GetSchedulerTypes().Contains(schedulerOptions.SchedulerType);
            if (!schedulerSupported)
                throw new Exception($"Scheduler '{schedulerOptions.SchedulerType}' is not compatible  with the `{pipeline.PipelineType}` pipeline.");

            return diffuser.DiffuseBatchAsync(modelOptions, promptOptions, schedulerOptions, batchOptions, progress, cancellationToken);
        }


    }
}
