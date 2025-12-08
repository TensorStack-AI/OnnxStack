using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Diffusers.StableDiffusionXL;
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
    public class StableDiffusionXLPipeline : StableDiffusionPipeline
    {
        protected ITokenizer _tokenizer2;
        protected TextEncoderModel _textEncoder2;

        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionXLPipeline"/> class.
        /// </summary>
        /// <param name="name">The pipeline name.</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="tokenizer2">The tokenizer2.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="textEncoder2">The text encoder2.</param>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public StableDiffusionXLPipeline(string name, ITokenizer tokenizer, ITokenizer tokenizer2, TextEncoderModel textEncoder, TextEncoderModel textEncoder2, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, UNetConditionModel controlNet, List<DiffuserType> diffusers, List<SchedulerType> schedulers = default, SchedulerOptions defaultSchedulerOptions = default, ILogger logger = default)
            : base(name, tokenizer, textEncoder, unet, vaeDecoder, vaeEncoder, controlNet, diffusers, schedulers, defaultSchedulerOptions, logger)
        {
            _tokenizer2 = tokenizer2;
            _textEncoder2 = textEncoder2;
            _supportedSchedulers = schedulers ?? new List<SchedulerType>
            {
                SchedulerType.LMS,
                SchedulerType.Euler,
                SchedulerType.EulerAncestral,
                SchedulerType.DDPM,
                SchedulerType.DDIM,
                SchedulerType.KDPM2,
                SchedulerType.KDPM2Ancestral,
                SchedulerType.LCM
            };
            _defaultSchedulerOptions = defaultSchedulerOptions ?? new SchedulerOptions
            {
                Width = 1024,
                Height = 1024,
                InferenceSteps = 28,
                GuidanceScale = 5f,
                SchedulerType = SchedulerType.EulerAncestral
            };
        }


        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override PipelineType PipelineType => PipelineType.StableDiffusionXL;


        /// <summary>
        /// Gets the tokenizer2.
        /// </summary>
        public ITokenizer Tokenizer2 => _tokenizer2;


        /// <summary>
        /// Gets the text encoder2.
        /// </summary>
        public TextEncoderModel TextEncoder2 => _textEncoder2;


        /// <summary>
        /// Unloads the pipeline.
        /// </summary>
        /// <returns></returns>
        public override Task UnloadAsync()
        {
            _textEncoder2?.Dispose();
            return base.UnloadAsync();
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
                DiffuserType.TextToImage => new TextDiffuser(_unet, _vaeDecoder, _vaeEncoder, _logger),
                DiffuserType.ImageToImage => new ImageDiffuser(_unet, _vaeDecoder, _vaeEncoder, _logger),
                DiffuserType.ImageInpaintLegacy => new InpaintLegacyDiffuser(_unet, _vaeDecoder, _vaeEncoder, _logger),
                DiffuserType.ControlNet => new ControlNetDiffuser(controlNetModel, _controlNetUnet, _vaeDecoder, _vaeEncoder, _logger),
                DiffuserType.ControlNetImage => new ControlNetImageDiffuser(controlNetModel, _controlNetUnet, _vaeDecoder, _vaeEncoder, _logger),
                _ => throw new NotImplementedException()
            };
        }


        /// <summary>
        /// Creates the prompt embeds.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns></returns>
        protected override async Task<PromptEmbeddingsResult> CreatePromptEmbedsAsync(GenerateOptions options, CancellationToken cancellationToken = default)
        {
            return _unet.ModelType switch
            {
                ModelType.Refiner => await CreateEmbedsTwoAsync(options, cancellationToken),
                _ => await CreateEmbedsBothAsync(options, cancellationToken),
            };
        }


        /// <summary>
        /// Checks the state of the pipeline.
        /// </summary>
        /// <param name="options">The options.</param>
        protected override async Task CheckPipelineState(GenerateOptions options)
        {
            await base.CheckPipelineState(options);
            if (options.IsLowMemoryTextEncoderEnabled && _textEncoder2?.Session is not null)
                await _textEncoder2.UnloadAsync();
        }


        /// <summary>
        /// Creates the embeds using Tokenizer2 and TextEncoder2
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns></returns>
        private async Task<PromptEmbeddingsResult> CreateEmbedsTwoAsync(GenerateOptions options, CancellationToken cancellationToken = default)
        {
            /// Tokenize Prompt and NegativePrompt with Tokenizer2
            var timestamp = _logger?.LogBegin();
            var promptTokens = await DecodeTextAsLongAsync(options.Prompt, cancellationToken);
            var negativePromptTokens = await DecodeTextAsLongAsync(options.NegativePrompt, cancellationToken);
            var maxPromptTokenCount = Math.Max(promptTokens.InputIds.Length, negativePromptTokens.InputIds.Length);

            // Generate embeds for tokens
            timestamp = _logger?.LogBegin();
            var promptEmbeddings = await GenerateEmbedsAsync(options, promptTokens, maxPromptTokenCount, cancellationToken);
            var negativePromptEmbeddings = await GenerateEmbedsAsync(options, negativePromptTokens, maxPromptTokenCount, cancellationToken);
            _logger?.LogEnd(LogLevel.Debug, $"TextEncoder2", timestamp);

            // Unload if required
            if (options.IsLowMemoryTextEncoderEnabled)
            {
                await _textEncoder2.UnloadAsync();
            }

            return new PromptEmbeddingsResult(promptEmbeddings.PromptEmbeds, promptEmbeddings.PooledPromptEmbeds, negativePromptEmbeddings.PromptEmbeds, negativePromptEmbeddings.PooledPromptEmbeds);
        }


        /// <summary>
        /// Creates the embeds using Tokenizer, Tokenizer2, TextEncoder and TextEncoder2
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns></returns>
        private async Task<PromptEmbeddingsResult> CreateEmbedsBothAsync(GenerateOptions options, CancellationToken cancellationToken = default)
        {
            // Tokenize Prompt and NegativePrompt with Tokenizer
            var timestamp = _logger?.LogBegin();
            var promptTokens = await DecodePromptTextAsync(options.Prompt, cancellationToken);
            var negativePromptTokens = await DecodePromptTextAsync(options.NegativePrompt, cancellationToken);
            var maxPromptTokenCount = Math.Max(promptTokens.InputIds.Length, negativePromptTokens.InputIds.Length);
            _logger?.LogEnd(LogLevel.Debug, $"Tokenizer", timestamp);

            // Generate embeds for tokens with TextEncoder
            timestamp = _logger?.LogBegin();
            var prompt1Embeddings = await GeneratePromptEmbedsAsync(promptTokens, maxPromptTokenCount, cancellationToken);
            var negativePrompt1Embeddings = await GeneratePromptEmbedsAsync(negativePromptTokens, maxPromptTokenCount, cancellationToken);
            _logger?.LogEnd(LogLevel.Debug, $"TextEncoder", timestamp);

            /// Tokenize Prompt and NegativePrompt with Tokenizer2
            timestamp = _logger?.LogBegin();
            var prompt2Tokens = await DecodeTextAsLongAsync(options.Prompt, cancellationToken);
            var negativePrompt2Tokens = await DecodeTextAsLongAsync(options.NegativePrompt, cancellationToken);
            _logger?.LogEnd(LogLevel.Debug, $"Tokenizer2", timestamp);

            // Generate embeds for tokens with TextEncoder2
            timestamp = _logger?.LogBegin();
            var prompt2Embeddings = await GenerateEmbedsAsync(options, prompt2Tokens, maxPromptTokenCount, cancellationToken);
            var negativePrompt2Embeddings = await GenerateEmbedsAsync(options, negativePrompt2Tokens, maxPromptTokenCount, cancellationToken);
            _logger?.LogEnd(LogLevel.Debug, $"TextEncoder2", timestamp);

            // Prompt embeds
            var pooledPromptEmbeds = prompt2Embeddings.PooledPromptEmbeds;
            var promptEmbeddings = prompt1Embeddings.PromptEmbeds.Concatenate(prompt2Embeddings.PromptEmbeds, 2);

            // Negative Prompt embeds
            var pooledNegativePromptEmbeds = negativePrompt2Embeddings.PooledPromptEmbeds;
            var negativePromptEmbeddings = negativePrompt1Embeddings.PromptEmbeds.Concatenate(negativePrompt2Embeddings.PromptEmbeds, 2);

            // Unload if required
            if (options.IsLowMemoryTextEncoderEnabled)
            {
                await _textEncoder.UnloadAsync();
                await _textEncoder2.UnloadAsync();
            }

            return new PromptEmbeddingsResult(promptEmbeddings, pooledPromptEmbeds, negativePromptEmbeddings, pooledNegativePromptEmbeds);
        }


        /// <summary>
        /// Decodes the text as tokens
        /// </summary>
        /// <param name="inputText">The input text.</param>
        /// <returns></returns>
        protected virtual async Task<TokenizerResult> DecodeTextAsLongAsync(string inputText, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(inputText))
                return new TokenizerResult([], []);

            return await _tokenizer2.EncodeAsync(inputText);
        }


        /// <summary>
        /// Encodes the tokens.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="tokenizedInput">The tokenized input.</param>
        /// <returns></returns>
        private async Task<EncoderResult> EncodeTokensAsync(GenerateOptions options, TokenizerResult tokenizedInput, CancellationToken cancellationToken = default)
        {
            var metadata = await _textEncoder2.LoadAsync(cancellationToken: cancellationToken);
            var inputTensor = new DenseTensor<long>(tokenizedInput.InputIds, [1, tokenizedInput.InputIds.Length]);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
            {
                int hiddenStateIndex = metadata.Outputs.Count - (2 + options.ClipSkip);
                inferenceParameters.AddInputTensor(inputTensor);

                // text_embeds + hidden_states.31 ("31" because SDXL always indexes from the penultimate layer.)
                inferenceParameters.AddOutputBuffer([1, _tokenizer2.TokenizerLength]);
                inferenceParameters.AddOutputBuffer(hiddenStateIndex, [1, tokenizedInput.InputIds.Length, _tokenizer2.TokenizerLength]);

                var results = await _textEncoder2.RunInferenceAsync(inferenceParameters);
                using (var promptEmbeds = results.Last())
                using (var promptEmbedsPooled = results.First())
                {
                    return new EncoderResult(promptEmbeds.ToDenseTensor(), promptEmbedsPooled.ToDenseTensor());
                }
            }
        }


        /// <summary>
        /// Generates the embeds.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="inputTokens">The input tokens.</param>
        /// <param name="minimumLength">The minimum length.</param>
        /// <returns></returns>
        protected async Task<EncoderResult> GenerateEmbedsAsync(GenerateOptions options, TokenizerResult inputTokens, int minimumLength, CancellationToken cancellationToken = default)
        {
            // If less than minimumLength pad with blank tokens
            if (inputTokens.InputIds.Length < minimumLength)
            {
                inputTokens.InputIds = PadWithBlankTokens(inputTokens.InputIds, minimumLength, _tokenizer.PadTokenId).ToArray();
                inputTokens.AttentionMask = PadWithBlankTokens(inputTokens.AttentionMask, minimumLength, 0).ToArray();
            }

            // The CLIP tokenizer only supports 77 tokens, batch process in groups of 77 and concatenate1
            var tokenBatches = new List<long[]>();
            var attentionBatches = new List<long[]>();
            foreach (var tokenBatch in inputTokens.InputIds.Chunk(_tokenizer.TokenizerLimit))
                tokenBatches.Add(PadWithBlankTokens(tokenBatch, _tokenizer.TokenizerLimit, _tokenizer.PadTokenId).ToArray());
            foreach (var attentionBatch in inputTokens.AttentionMask.Chunk(_tokenizer.TokenizerLimit))
                attentionBatches.Add(PadWithBlankTokens(attentionBatch, _tokenizer.TokenizerLimit, 0).ToArray());

            var promptEmbeddings = new List<float>();
            var pooledPromptEmbeddings = new List<float>();
            for (int i = 0; i < tokenBatches.Count; i++)
            {
                var result = await EncodeTokensAsync(options, new TokenizerResult(tokenBatches[i], attentionBatches[i]), cancellationToken);
                promptEmbeddings.AddRange(result.PromptEmbeds);
                pooledPromptEmbeddings.AddRange(result.PooledPromptEmbeds);
            }

            var promptTensor = new DenseTensor<float>(promptEmbeddings.ToArray(), [1, promptEmbeddings.Count / _tokenizer2.TokenizerLength, _tokenizer2.TokenizerLength]);
            var pooledTensor = new DenseTensor<float>(pooledPromptEmbeddings.Take(_tokenizer2.TokenizerLength).ToArray(), [1, _tokenizer2.TokenizerLength]);
            return new EncoderResult(promptTensor, pooledTensor);
        }


        /// <summary>
        /// Creates the pipeline from a ModelSet configuration.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static new StableDiffusionXLPipeline CreatePipeline(StableDiffusionModelSet modelSet, ILogger logger = default)
        {
            var config = modelSet with { };
            var unet = new UNetConditionModel(config.UnetConfig.ApplyDefaults(config));
            var tokenizer = new ClipTokenizer(config.TokenizerConfig.ApplyDefaults(config));
            var tokenizer2 = new ClipTokenizer(config.Tokenizer2Config.ApplyDefaults(config));
            var textEncoder = new TextEncoderModel(config.TextEncoderConfig.ApplyDefaults(config));
            var textEncoder2 = new TextEncoderModel(config.TextEncoder2Config.ApplyDefaults(config));
            var vaeDecoder = new AutoEncoderModel(config.VaeDecoderConfig.ApplyDefaults(config));
            var vaeEncoder = new AutoEncoderModel(config.VaeEncoderConfig.ApplyDefaults(config));
            var controlnet = default(UNetConditionModel);
            if (config.ControlNetUnetConfig is not null)
                controlnet = new UNetConditionModel(config.ControlNetUnetConfig.ApplyDefaults(config));

            LogPipelineInfo(modelSet, logger);
            return new StableDiffusionXLPipeline(config.Name, tokenizer, tokenizer2, textEncoder, textEncoder2, unet, vaeDecoder, vaeEncoder, controlnet, config.Diffusers, config.Schedulers, config.SchedulerOptions, logger);
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
        public static new StableDiffusionXLPipeline CreatePipeline(OnnxExecutionProvider executionProvider, string modelFolder, ModelType modelType = ModelType.Base, ILogger logger = default)
        {
            return CreatePipeline(ModelFactory.CreateStableDiffusionXLModelSet(modelFolder, modelType).WithProvider(executionProvider), logger);
        }
    }
}
