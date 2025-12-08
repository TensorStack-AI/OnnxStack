using Microsoft.Extensions.Logging;
using OnnxStack.Core;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Diffusers.Locomotion;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.StableDiffusion.Tokenizers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Pipelines
{
    public sealed class LocomotionPipeline : StableDiffusionPipeline
    {
        private readonly ResampleModel _resampler;
        private readonly FlowEstimationModel _flowEstimation;

        /// <summary>
        /// Initializes a new instance of the <see cref="LocomotionPipeline"/> class.
        /// </summary>
        /// <param name="name">The pipeline name.</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public LocomotionPipeline(string name, ITokenizer tokenizer, TextEncoderModel textEncoder, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, UNetConditionModel controlNet, FlowEstimationModel flowEstimation, ResampleModel resampler, List<DiffuserType> diffusers, List<SchedulerType> schedulers, SchedulerOptions defaultSchedulerOptions = default, ILogger logger = default)
            : base(name, tokenizer, textEncoder, unet, vaeDecoder, vaeEncoder, controlNet, diffusers, schedulers, defaultSchedulerOptions, logger)
        {
            _resampler = resampler;
            _flowEstimation = flowEstimation;
            _supportedDiffusers = diffusers ?? new List<DiffuserType>
            {
                DiffuserType.TextToVideo,
                DiffuserType.ImageToVideo,
                DiffuserType.VideoToVideo,
                DiffuserType.ControlNet,
                DiffuserType.ControlNetImage,
                DiffuserType.ControlNetVideo
            };
            _supportedSchedulers = schedulers ?? new List<SchedulerType>
            {
                SchedulerType.Locomotion
            };
            _defaultSchedulerOptions = defaultSchedulerOptions ?? new SchedulerOptions
            {
                Width = 512,
                Height = 512,
                GuidanceScale = 0f,
                InferenceSteps = 8,
                SchedulerType = SchedulerType.Locomotion,
                BetaSchedule = BetaScheduleType.Linear
            };
        }


        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override PipelineType PipelineType => PipelineType.Locomotion;


        /// <summary>
        /// Unloads the pipeline.
        /// </summary>
        /// <returns>A Task representing the asynchronous operation.</returns>
        public override async Task UnloadAsync()
        {
            await _resampler.UnloadAsync();
            await _flowEstimation.UnloadAsync();
            await base.UnloadAsync();
        }


        /// <summary>
        /// Creates the diffuser.
        /// </summary>
        /// <param name="diffuserType">Type of the diffuser.</param>
        /// <param name="controlNetModel">The control net model.</param>
        /// <returns></returns>
        protected override IDiffuser CreateDiffuser(DiffuserType diffuserType, ControlNetModel controlNetModel)
        {
            return diffuserType switch
            {
                DiffuserType.TextToVideo => new TextDiffuser(_unet, _vaeDecoder, _vaeEncoder, _flowEstimation, _resampler, _logger),
                DiffuserType.ImageToVideo => new ImageDiffuser(_unet, _vaeDecoder, _vaeEncoder, _flowEstimation, _resampler, _logger),
                DiffuserType.VideoToVideo => new VideoDiffuser(_unet, _vaeDecoder, _vaeEncoder, _flowEstimation, _resampler, _logger),
                DiffuserType.ControlNet => new ControlNetDiffuser(controlNetModel, _controlNetUnet, _vaeDecoder, _vaeEncoder, _flowEstimation, _resampler, _logger),
                DiffuserType.ControlNetImage => new ControlNetImageDiffuser(controlNetModel, _controlNetUnet, _vaeDecoder, _vaeEncoder, _flowEstimation, _resampler, _logger),
                DiffuserType.ControlNetVideo => new ControlNetVideoDiffuser(controlNetModel, _controlNetUnet, _vaeDecoder, _vaeEncoder, _flowEstimation, _resampler, _logger),
                _ => throw new NotImplementedException()
            };
        }


        /// <summary>
        /// Checks the state of the pipeline.
        /// </summary>
        /// <param name="options">The options.</param>
        protected override async Task CheckPipelineState(GenerateOptions options)
        {
            await base.CheckPipelineState(options);
            if (options.IsLowMemoryDecoderEnabled && _flowEstimation?.Session is not null)
                await _flowEstimation.UnloadAsync();
            if (options.IsLowMemoryDecoderEnabled && _resampler?.Session is not null)
                await _resampler.UnloadAsync();
        }


        /// <summary>
        /// Creates the prompt embeds.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns></returns>
        protected override async Task<PromptEmbeddingsResult> CreatePromptEmbedsAsync(GenerateOptions options, CancellationToken cancellationToken = default)
        {
            // Repeat prompt per frame
            var frameCount = (int)Math.Ceiling(options.FrameCount / (double)_unet.ContextSize) * _unet.ContextSize;
            if (options.Prompts.IsNullOrEmpty())
            {
                var originalPromptEmbeds = await CreatePromptEmbedsAsync(options.Prompt, options.NegativePrompt, cancellationToken);
                var promptEmbeds = originalPromptEmbeds.PromptEmbeds.Repeat(frameCount);
                var pooledPromptEmbeds = originalPromptEmbeds.PooledPromptEmbeds.Repeat(frameCount);
                var negativePromptEmbeds = originalPromptEmbeds.NegativePromptEmbeds.Repeat(frameCount);
                var negativePooledPromptEmbeds = originalPromptEmbeds.NegativePooledPromptEmbeds.Repeat(frameCount);
                if (options.IsLowMemoryTextEncoderEnabled)
                    await _textEncoder.UnloadAsync();

                return new PromptEmbeddingsResult(promptEmbeds, pooledPromptEmbeds, negativePromptEmbeds, negativePooledPromptEmbeds);
            }
            else
            {
                var results = new List<PromptEmbeddingsResult>();
                var prompts = options.Prompts.Take(frameCount).ToList();
                var repeatCount = frameCount / prompts.Count;
                foreach (var prompt in prompts)
                {
                    var embeds = await CreatePromptEmbedsAsync(prompt, options.NegativePrompt, cancellationToken);
                    for (int i = 0; i < repeatCount; i++)
                        results.Add(embeds);
                }

                while (results.Count < frameCount)
                    results.Insert(0, results.First());

                var promptEmbeds = results.Select(x => x.PromptEmbeds).ToArray().Join();
                var pooledPromptEmbeds = results.Select(x => x.PooledPromptEmbeds).ToArray().Join();
                var negativePromptEmbeds = results.Select(x => x.NegativePromptEmbeds).ToArray().Join();
                var negativePooledPromptEmbeds = results.Select(x => x.NegativePooledPromptEmbeds).ToArray().Join();
                if (options.IsLowMemoryTextEncoderEnabled)
                    await _textEncoder.UnloadAsync();

                return new PromptEmbeddingsResult(promptEmbeds, pooledPromptEmbeds, negativePromptEmbeds, negativePooledPromptEmbeds);
            }
        }


        /// <summary>
        /// Create prompt embeds
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="negativePrompt">The negative prompt.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;PromptEmbeddingsResult&gt; representing the asynchronous operation.</returns>
        private async Task<PromptEmbeddingsResult> CreatePromptEmbedsAsync(string prompt, string negativePrompt, CancellationToken cancellationToken = default)
        {
            var promptTokens = await DecodePromptTextAsync(prompt, cancellationToken);
            var negativePromptTokens = await DecodePromptTextAsync(negativePrompt, cancellationToken);
            var maxPromptTokenCount = Math.Max(promptTokens.InputIds.Length, negativePromptTokens.InputIds.Length);

            var promptEmbeddings = await GeneratePromptEmbedsAsync(promptTokens, maxPromptTokenCount, cancellationToken);
            var negativePromptEmbeddings = await GeneratePromptEmbedsAsync(negativePromptTokens, maxPromptTokenCount, cancellationToken);
            return new PromptEmbeddingsResult(promptEmbeddings.PromptEmbeds, promptEmbeddings.PooledPromptEmbeds, negativePromptEmbeddings.PromptEmbeds, negativePromptEmbeddings.PooledPromptEmbeds);
        }


        /// <summary>
        /// Creates the pipeline from a ModelSet configuration.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static new LocomotionPipeline CreatePipeline(StableDiffusionModelSet modelSet, ILogger logger = default)
        {
            var config = modelSet with { };
            var unet = new UNetConditionModel(config.UnetConfig.ApplyDefaults(config));
            var tokenizer = new ClipTokenizer(config.TokenizerConfig.ApplyDefaults(config));
            var textEncoder = new TextEncoderModel(config.TextEncoderConfig.ApplyDefaults(config));
            var vaeDecoder = new AutoEncoderModel(config.VaeDecoderConfig.ApplyDefaults(config));
            var vaeEncoder = new AutoEncoderModel(config.VaeEncoderConfig.ApplyDefaults(config));
            var resampler = new ResampleModel(config.ResampleModelConfig.ApplyDefaults(config));
            var flowEstimation = new FlowEstimationModel(config.FlowEstimationConfig.ApplyDefaults(config));
            var controlnet = default(UNetConditionModel);
            if (config.ControlNetUnetConfig is not null)
                controlnet = new UNetConditionModel(config.ControlNetUnetConfig.ApplyDefaults(config));

            LogPipelineInfo(modelSet, logger);
            return new LocomotionPipeline(config.Name, tokenizer, textEncoder, unet, vaeDecoder, vaeEncoder, controlnet, flowEstimation, resampler, config.Diffusers, config.Schedulers, config.SchedulerOptions, logger);
        }


        /// <summary>
        /// Creates the pipeline from a folder structure.
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static LocomotionPipeline CreatePipeline(OnnxExecutionProvider executionProvider, string modelFolder, int contextSize, ModelType modelType = ModelType.Base, ILogger logger = default)
        {
            return CreatePipeline(ModelFactory.CreateLocomotionModelSet(modelFolder, contextSize).WithProvider(executionProvider), logger);
        }
    }
}
