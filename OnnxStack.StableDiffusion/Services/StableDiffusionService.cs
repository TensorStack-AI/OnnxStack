using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Helpers;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Generic;
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
        private readonly IDiffuser _textDiffuser;
        private readonly IDiffuser _imageDiffuser;
        private readonly IDiffuser _inpaintDiffuser;
        private readonly IDiffuser _inpaintLegacyDiffuser;
        private readonly IOnnxModelService _onnxModelService;
        private readonly StableDiffusionConfig _configuration;


        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionService"/> class.
        /// </summary>
        /// <param name="schedulerService">The scheduler service.</param>
        public StableDiffusionService(StableDiffusionConfig configuration, IOnnxModelService onnxModelService, IPromptService promptService)
        {
            _configuration = configuration;
            _onnxModelService = onnxModelService;
            _textDiffuser = new TextDiffuser(onnxModelService, promptService);
            _imageDiffuser = new ImageDiffuser(onnxModelService, promptService);
            _inpaintDiffuser = new InpaintDiffuser(onnxModelService, promptService);
            _inpaintLegacyDiffuser = new InpaintLegacyDiffuser(onnxModelService, promptService);
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
            return await RunAsync(model, prompt, options, progressCallback, cancellationToken).ConfigureAwait(false);
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


        private async Task<DenseTensor<float>> RunAsync(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, Action<int, int> progress = null, CancellationToken cancellationToken = default)
        {
            return promptOptions.ProcessType switch
            {
                ProcessType.TextToImage => await _textDiffuser.DiffuseAsync(modelOptions, promptOptions, schedulerOptions, progress, cancellationToken),
                ProcessType.ImageToImage => await _imageDiffuser.DiffuseAsync(modelOptions, promptOptions, schedulerOptions, progress, cancellationToken),
                ProcessType.ImageInpaint => await _inpaintLegacyDiffuser.DiffuseAsync(modelOptions, promptOptions, schedulerOptions, progress, cancellationToken),
                _ => throw new NotImplementedException()
            };
        }
    }
}
