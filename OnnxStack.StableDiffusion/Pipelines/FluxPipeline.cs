using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Diffusers.Flux;
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
    public sealed class FluxPipeline : StableDiffusionXLPipeline
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusion3Pipeline"/> class.
        /// </summary>
        /// <param name="name">The pipeline name</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="tokenizer2">The tokenizer2.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="textEncoder2">The text encoder2.</param>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public FluxPipeline(string name, ITokenizer tokenizer, ITokenizer tokenizer2, TextEncoderModel textEncoder, TextEncoderModel textEncoder2, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, UNetConditionModel controlNet, List<DiffuserType> diffusers, List<SchedulerType> schedulers, SchedulerOptions defaultSchedulerOptions = default, ILogger logger = default)
            : base(name, tokenizer, tokenizer2, textEncoder, textEncoder2, unet, vaeDecoder, vaeEncoder, controlNet, diffusers, schedulers, defaultSchedulerOptions, logger)
        {
            _supportedSchedulers = schedulers ?? new List<SchedulerType>
            {
                SchedulerType.FlowMatchEulerDiscrete,
                SchedulerType.FlowMatchEulerDynamic
            };
            _defaultSchedulerOptions = defaultSchedulerOptions ?? new SchedulerOptions
            {
                Width = 1024,
                Height = 1024,
                GuidanceScale = 0,
                TimestepSpacing = TimestepSpacingType.Leading,
                InferenceSteps = tokenizer2.TokenizerLimit == 256 ? 4 : 28,
                GuidanceScale2 = tokenizer2.TokenizerLimit == 256 ? 0 : 3.5f,
                SchedulerType = SchedulerType.FlowMatchEulerDiscrete
            };
        }

        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override PipelineType PipelineType => PipelineType.Flux;


        /// <summary>
        /// Unloads the pipeline.
        /// </summary>
        /// <returns>A Task representing the asynchronous operation.</returns>
        public override async Task UnloadAsync()
        {
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
            // Flux Kontext
            if (_unet.ModelType == ModelType.Instruct)
                return new InstructDiffuser(_unet, _vaeDecoder, _vaeEncoder, _logger);

            return diffuserType switch
            {
                DiffuserType.TextToImage => new TextDiffuser(_unet, _vaeDecoder, _vaeEncoder, _logger),
                DiffuserType.ImageToImage => new ImageDiffuser(_unet, _vaeDecoder, _vaeEncoder, _logger),
                _ => throw new NotImplementedException()
            };
        }

        /// <summary>
        /// Creates the prompt embeds.
        /// </summary>
        /// <param name="options">The prompt options.</param>
        /// <returns></returns>
        protected override async Task<PromptEmbeddingsResult> CreatePromptEmbedsAsync(GenerateOptions options, CancellationToken cancellationToken = default)
        {
            // Tokenize Prompt
            var promptTokens = await DecodePromptTextAsync(options.Prompt, cancellationToken);
            var negativePromptTokens = await DecodePromptTextAsync(options.NegativePrompt, cancellationToken);
            var maxTokenLength = Math.Max(promptTokens.InputIds.Length, negativePromptTokens.InputIds.Length);

            // Generate embeds for tokens
            var promptEmbeddings = await GeneratePromptEmbedsAsync(promptTokens, maxTokenLength, cancellationToken);
            var negativePromptEmbeddings = await GeneratePromptEmbedsAsync(negativePromptTokens, maxTokenLength, cancellationToken);

            /// Tokenize Prompt with Tokenizer2
            var prompt2Tokens = await DecodeTextAsLongAsync(options.Prompt, cancellationToken);
            var negativePrompt2Tokens = await DecodeTextAsLongAsync(options.NegativePrompt, cancellationToken);

            // Generate embeds for tokens with TextEncoder2
            var prompt2Embeddings = await GeneratePrompt2EmbedsAsync(prompt2Tokens, _tokenizer2.TokenizerLimit, cancellationToken);
            var negativePrompt2Embeddings = await GeneratePrompt2EmbedsAsync(negativePrompt2Tokens, _tokenizer2.TokenizerLimit, cancellationToken);

            // We batch CLIP greater than 77 not truncate so the pool embeds wont work that way, so only use first set
            var promptPooledEmbeds = promptEmbeddings.PooledPromptEmbeds
                .ReshapeTensor([promptEmbeddings.PooledPromptEmbeds.Dimensions[^2], promptEmbeddings.PooledPromptEmbeds.Dimensions[^1]])
                .FirstBatch();
            var negativePromptPooledEmbeds = negativePromptEmbeddings.PooledPromptEmbeds
                .ReshapeTensor([negativePromptEmbeddings.PooledPromptEmbeds.Dimensions[^2], negativePromptEmbeddings.PooledPromptEmbeds.Dimensions[^1]])
                .FirstBatch();

            // Unload if required
            if (options.IsLowMemoryTextEncoderEnabled)
            {
                await _textEncoder.UnloadAsync();
                await _textEncoder2.UnloadAsync();
            }

            return new PromptEmbeddingsResult(prompt2Embeddings, promptPooledEmbeds, negativePrompt2Embeddings, negativePromptPooledEmbeds);
        }


        /// <summary>
        /// Generates the prompt2 embeds asynchronous.
        /// </summary>
        /// <param name="prompt3Tokens">The prompt3 tokens.</param>
        /// <param name="maxPromptTokenCount">The maximum prompt token count.</param>
        /// <returns></returns>
        private async Task<DenseTensor<float>> GeneratePrompt2EmbedsAsync(TokenizerResult promptTokens, int maxPromptTokenCount, CancellationToken cancellationToken = default)
        {
            if (promptTokens is null || promptTokens.InputIds.IsNullOrEmpty())
                return new DenseTensor<float>([1, maxPromptTokenCount, _tokenizer2.TokenizerLength]);

            var inputIds = PadWithBlankTokens(promptTokens.InputIds, maxPromptTokenCount, 0).ToArray().ToInt();

            var metadata = await _textEncoder2.LoadAsync(cancellationToken: cancellationToken);
            var inputTensor = new DenseTensor<int>(inputIds, [1, inputIds.Length]);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
            {
                inferenceParameters.AddInputTensor(inputTensor);
                inferenceParameters.AddOutputBuffer([1, maxPromptTokenCount, _tokenizer2.TokenizerLength]);

                var results = await _textEncoder2.RunInferenceAsync(inferenceParameters);
                using (var result = results.First())
                {
                    return result.ToDenseTensor();
                }
            }
        }


        /// <summary>
        /// Creates the pipeline from a ModelSet configuration.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static new FluxPipeline CreatePipeline(StableDiffusionModelSet modelSet, ILogger logger = default)
        {
            var config = modelSet with { };
            var unet = new UNetConditionModel(config.UnetConfig.ApplyDefaults(config));
            var tokenizer = new ClipTokenizer(config.TokenizerConfig.ApplyDefaults(config));
            var tokenizer2 = new SentencePieceTokenizer(config.Tokenizer2Config.ApplyDefaults(config));
            var textEncoder = new TextEncoderModel(config.TextEncoderConfig.ApplyDefaults(config));
            var textEncoder2 = new TextEncoderModel(config.TextEncoder2Config.ApplyDefaults(config));
            var vaeDecoder = new AutoEncoderModel(config.VaeDecoderConfig.ApplyDefaults(config));
            var vaeEncoder = new AutoEncoderModel(config.VaeEncoderConfig.ApplyDefaults(config));
            var controlnet = default(UNetConditionModel);
            if (config.ControlNetUnetConfig is not null)
                controlnet = new UNetConditionModel(config.ControlNetUnetConfig.ApplyDefaults(config));

            LogPipelineInfo(modelSet, logger);
            return new FluxPipeline(config.Name, tokenizer, tokenizer2, textEncoder, textEncoder2, unet, vaeDecoder, vaeEncoder, controlnet, config.Diffusers, config.Schedulers, config.SchedulerOptions, logger);
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
        public static new FluxPipeline CreatePipeline(OnnxExecutionProvider executionProvider, string modelFolder, ModelType modelType = ModelType.Base, ILogger logger = default)
        {
            return CreatePipeline(ModelFactory.CreateFluxModelSet(modelFolder, modelType).WithProvider(executionProvider), logger);
        }
    }
}
