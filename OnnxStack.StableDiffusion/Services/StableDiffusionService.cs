using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
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
        private readonly IOnnxModelService _onnxModelService;
        private readonly StableDiffusionConfig _configuration;
        private readonly IDictionary<DiffuserPipelineType, IDictionary<DiffuserType, IDiffuser>> _diffusers;

        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionService"/> class.
        /// </summary>
        /// <param name="schedulerService">The scheduler service.</param>
        public StableDiffusionService(StableDiffusionConfig configuration, IOnnxModelService onnxModelService, IPromptService promptService)
        {
            _configuration = configuration;
            _onnxModelService = onnxModelService;
            var stableDiffusionPipeline = new Dictionary<DiffuserType, IDiffuser>
            {
                { DiffuserType.TextToImage, new TextDiffuser(onnxModelService, promptService) },
                { DiffuserType.ImageToImage, new ImageDiffuser(onnxModelService, promptService) },
                { DiffuserType.ImageInpaint, new InpaintDiffuser(onnxModelService, promptService) },
                { DiffuserType.ImageInpaintLegacy, new InpaintLegacyDiffuser(onnxModelService, promptService) }
            };

            var latentConsistancyPipeline = new Dictionary<DiffuserType, IDiffuser>
            {
                //TODO: TextToImage and ImageToImage is supported with LCM
            };

            _diffusers.Add(DiffuserPipelineType.StableDiffusion, stableDiffusionPipeline);
            _diffusers.Add(DiffuserPipelineType.LatentConsistency, latentConsistancyPipeline);
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


        private async Task<DenseTensor<float>> DiffuseAsync(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, Action<int, int> progress = null, CancellationToken cancellationToken = default)
        {
            var pipeline = _diffusers[modelOptions.PipelineType];
            if (pipeline is null)
                throw new Exception("Pipeline not found or is unsupported");

            var diffuser = pipeline[promptOptions.DiffuserType];
            if (diffuser is null)
                throw new Exception("Diffuser not found or is unsupported");

            return await diffuser.DiffuseAsync(modelOptions, promptOptions, schedulerOptions, progress, cancellationToken);
        }
    }
}
