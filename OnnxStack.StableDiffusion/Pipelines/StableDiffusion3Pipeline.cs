using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Diffusers.StableDiffusion3;
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
    public class StableDiffusion3Pipeline : StableDiffusionXLPipeline
    {
        private readonly ITokenizer _tokenizer3;
        private readonly TextEncoderModel _textEncoder3;

        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusion3Pipeline"/> class.
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
        public StableDiffusion3Pipeline(string name, ITokenizer tokenizer, ITokenizer tokenizer2, ITokenizer tokenizer3, TextEncoderModel textEncoder, TextEncoderModel textEncoder2, TextEncoderModel textEncoder3, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, UNetConditionModel controlNet, List<DiffuserType> diffusers, List<SchedulerType> schedulers, SchedulerOptions defaultSchedulerOptions = default, ILogger logger = default)
            : base(name, tokenizer, tokenizer2, textEncoder, textEncoder2, unet, vaeDecoder, vaeEncoder, controlNet, diffusers, schedulers, defaultSchedulerOptions, logger)
        {
            _tokenizer3 = tokenizer3;
            _textEncoder3 = textEncoder3;
            _supportedSchedulers = schedulers ?? new List<SchedulerType>
            {
                SchedulerType.FlowMatchEulerDiscrete,
                SchedulerType.FlowMatchEulerDynamic
            };
            _defaultSchedulerOptions = defaultSchedulerOptions ?? new SchedulerOptions
            {
                Shift = 3f,
                Width = 1024,
                Height = 1024,
                InferenceSteps = 28,
                GuidanceScale = 3f,
                SchedulerType = SchedulerType.FlowMatchEulerDiscrete
            };
        }

        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override PipelineType PipelineType => PipelineType.StableDiffusion3;


        /// <summary>
        /// Unloads the pipeline.
        /// </summary>
        /// <returns></returns>
        public override Task UnloadAsync()
        {
            _tokenizer3?.Dispose();
            _textEncoder3?.Dispose();
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
                DiffuserType.ControlNet => new ControlNetDiffuser(controlNetModel, _controlNetUnet, _vaeDecoder, _vaeEncoder, _logger),
                DiffuserType.ControlNetImage => new ControlNetImageDiffuser(controlNetModel, _controlNetUnet, _vaeDecoder, _vaeEncoder, _logger),
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
            if (options.IsLowMemoryTextEncoderEnabled && _textEncoder3?.Session is not null)
                await _textEncoder3.UnloadAsync();
        }


        /// <summary>
        /// Creates the prompt embeds.
        /// </summary>
        /// <param name="options">The prompt options.</param>
        /// <param name="isGuidanceEnabled">if set to <c>true</c> [is guidance enabled].</param>
        /// <returns></returns>
        protected override async Task<PromptEmbeddingsResult> CreatePromptEmbedsAsync(GenerateOptions options, CancellationToken cancellationToken = default)
        {
            // Tokenize Prompt and NegativePrompt
            var timestamp = _logger?.LogBegin();

            await Task.WhenAll
            (
                _textEncoder.LoadAsync(cancellationToken: cancellationToken),
                _textEncoder2.LoadAsync(cancellationToken: cancellationToken),
                _textEncoder3?.LoadAsync(cancellationToken: cancellationToken) ?? Task.CompletedTask
            );

            var promptTokens = await DecodePromptTextAsync(options.Prompt, cancellationToken);
            var negativePromptTokens = await DecodePromptTextAsync(options.NegativePrompt, cancellationToken);
            var maxPromptTokenCount = Math.Max(promptTokens.InputIds.Length, negativePromptTokens.InputIds.Length);
            _logger?.LogEnd(LogLevel.Debug, $"Tokenizer", timestamp);

            // Generate embeds for tokens
            timestamp = _logger?.LogBegin();
            var promptEmbeddings = await GeneratePromptEmbedsAsync(promptTokens, maxPromptTokenCount, cancellationToken);
            var negativePromptEmbeddings = await GeneratePromptEmbedsAsync(negativePromptTokens, maxPromptTokenCount, cancellationToken);
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

            /// Tokenize Prompt and NegativePrompt with Tokenizer3
            timestamp = _logger?.LogBegin();
            var prompt3Tokens = await TokenizePrompt3Async(options.Prompt, cancellationToken);
            var negativePrompt3Tokens = await TokenizePrompt3Async(options.NegativePrompt, cancellationToken);
            _logger?.LogEnd(LogLevel.Debug, $"Tokenizer3", timestamp);

            // Generate embeds for tokens with TextEncoder3
            timestamp = _logger?.LogBegin();
            var prompt3Embeddings = await GeneratePrompt3EmbedsAsync(prompt3Tokens, _tokenizer3.TokenizerLimit, cancellationToken);
            var negativePrompt3Embeddings = await GeneratePrompt3EmbedsAsync(negativePrompt3Tokens, _tokenizer3.TokenizerLimit, cancellationToken);
            _logger?.LogEnd(LogLevel.Debug, $"TextEncoder3", timestamp);


            // Positive Prompt
            var prompt_embed = promptEmbeddings.PromptEmbeds;

            // We batch CLIP greater than 77 not truncate so the pool embeds wont work that way, so only use first set
            var pooled_prompt_embed = promptEmbeddings.PooledPromptEmbeds
                .ReshapeTensor([promptEmbeddings.PooledPromptEmbeds.Dimensions[^2], promptEmbeddings.PooledPromptEmbeds.Dimensions[^1]])
                .FirstBatch();

            var prompt_2_embed = prompt2Embeddings.PromptEmbeds;
            var pooled_prompt_2_embed = prompt2Embeddings.PooledPromptEmbeds.FirstBatch();

            var clip_prompt_embeds = prompt_embed.Concatenate(prompt_2_embed, 2);
            clip_prompt_embeds = clip_prompt_embeds.PadEnd(prompt3Embeddings.Dimensions[^1] - clip_prompt_embeds.Dimensions[^1]);
            var prompt_embeds = clip_prompt_embeds.Concatenate(prompt3Embeddings, 1);

            pooled_prompt_2_embed = pooled_prompt_2_embed.Repeat(pooled_prompt_embed.Dimensions[0]);
            var pooled_prompt_embeds = pooled_prompt_embed.Concatenate(pooled_prompt_2_embed, 1);


            // Negative Prompt
            var negative_prompt_embed = negativePromptEmbeddings.PromptEmbeds;

            // We batch CLIP greater than 77 not truncate so the pool embeds wont work that way, so only use first set
            var negative_pooled_prompt_embed = negativePromptEmbeddings.PooledPromptEmbeds
                .ReshapeTensor([negativePromptEmbeddings.PooledPromptEmbeds.Dimensions[^2], negativePromptEmbeddings.PooledPromptEmbeds.Dimensions[^1]])
                .FirstBatch();

            var negative_prompt_2_embed = negativePrompt2Embeddings.PromptEmbeds;
            var negative_pooled_prompt_2_embed = negativePrompt2Embeddings.PooledPromptEmbeds.FirstBatch();

            var negative_clip_prompt_embeds = negative_prompt_embed.Concatenate(negative_prompt_2_embed, 2);
            negative_clip_prompt_embeds = negative_clip_prompt_embeds.PadEnd(negativePrompt3Embeddings.Dimensions[^1] - negative_clip_prompt_embeds.Dimensions[^1]);
            var negative_prompt_embeds = negative_clip_prompt_embeds.Concatenate(negativePrompt3Embeddings, 1);

            negative_pooled_prompt_2_embed = negative_pooled_prompt_2_embed.Repeat(negative_pooled_prompt_embed.Dimensions[0]);
            var negative_pooled_prompt_embeds = negative_pooled_prompt_embed.Concatenate(negative_pooled_prompt_2_embed, 1);


            // Unload if required
            if (options.IsLowMemoryTextEncoderEnabled)
            {
                await _textEncoder.UnloadAsync();
                await _textEncoder2.UnloadAsync();
                if (_textEncoder3 is not null)
                    await _textEncoder3.UnloadAsync();
            }

            return new PromptEmbeddingsResult(prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds);
        }


        /// <summary>
        /// Generates the prompt3 embeds asynchronous.
        /// </summary>
        /// <param name="prompt3Tokens">The prompt3 tokens.</param>
        /// <param name="maxPromptTokenCount">The maximum prompt token count.</param>
        /// <returns></returns>
        protected virtual async Task<DenseTensor<float>> GeneratePrompt3EmbedsAsync(TokenizerResult prompt3Tokens, int maxPromptTokenCount, CancellationToken cancellationToken = default)
        {
            if (prompt3Tokens is null || prompt3Tokens.InputIds.IsNullOrEmpty())
                return new DenseTensor<float>([1, maxPromptTokenCount, 4096]);

            var inputIds = PadWithBlankTokens(prompt3Tokens.InputIds, maxPromptTokenCount, 0).ToArray();

            var metadata = await _textEncoder3.LoadAsync(cancellationToken: cancellationToken);
            var inputTensor = GetTextEncoderInputTensor(metadata.Inputs[0], inputIds);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
            {
                inferenceParameters.AddInput(inputTensor);
                inferenceParameters.AddOutputBuffer();

                using (var results = _textEncoder3.RunInference(inferenceParameters))
                {
                    return results[0].ToDenseTensor();
                }
            }
        }


        /// <summary>
        /// Gets the text encoder input tensor.
        /// </summary>
        /// <param name="metadata">The metadata.</param>
        /// <param name="inputIds">The input ids.</param>
        /// <returns>Microsoft.ML.OnnxRuntime.OrtValue.</returns>
        private OrtValue GetTextEncoderInputTensor(OnnxNamedMetadata metadata, long[] inputIds)
        {
            if (metadata.Value.ElementDataType == TensorElementType.Int32)
                return new DenseTensor<int>(inputIds.ToInt(), [1, inputIds.Length]).ToOrtValue(metadata);

            return new DenseTensor<long>(inputIds, [1, inputIds.Length]).ToOrtValue(metadata);
        }


        /// <summary>
        /// Decodes the text as tokens
        /// </summary>
        /// <param name="inputText">The input text.</param>
        /// <returns></returns>
        private async Task<TokenizerResult> TokenizePrompt3Async(string inputText, CancellationToken cancellationToken = default)
        {
            if (_tokenizer3 is null)
                return null;

            if (string.IsNullOrEmpty(inputText))
                return new TokenizerResult([], []);

            return await _tokenizer3.EncodeAsync(inputText);
        }


        /// <summary>
        /// Creates the pipeline from a ModelSet configuration.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static new StableDiffusion3Pipeline CreatePipeline(StableDiffusionModelSet modelSet, ILogger logger = default)
        {
            var config = modelSet with { };
            var unet = new UNetConditionModel(config.UnetConfig.ApplyDefaults(config));
            var tokenizer = new ClipTokenizer(config.TokenizerConfig.ApplyDefaults(config));
            var tokenizer2 = new ClipTokenizer(config.Tokenizer2Config.ApplyDefaults(config));
            var textEncoder = new TextEncoderModel(config.TextEncoderConfig.ApplyDefaults(config));
            var textEncoder2 = new TextEncoderModel(config.TextEncoder2Config.ApplyDefaults(config));
            var vaeDecoder = new AutoEncoderModel(config.VaeDecoderConfig.ApplyDefaults(config));
            var vaeEncoder = new AutoEncoderModel(config.VaeEncoderConfig.ApplyDefaults(config));

            var tokenizer3 = default(ITokenizer);
            if (config.Tokenizer3Config is not null)
                tokenizer3 = new SentencePieceTokenizer(config.Tokenizer3Config.ApplyDefaults(config));

            var textEncoder3 = default(TextEncoderModel);
            if (config.TextEncoder3Config is not null)
                textEncoder3 = new TextEncoderModel(config.TextEncoder3Config.ApplyDefaults(config));

            var controlnet = default(UNetConditionModel);
            if (config.ControlNetUnetConfig is not null)
                controlnet = new UNetConditionModel(config.ControlNetUnetConfig.ApplyDefaults(config));

            LogPipelineInfo(modelSet, logger);
            return new StableDiffusion3Pipeline(config.Name, tokenizer, tokenizer2, tokenizer3, textEncoder, textEncoder2, textEncoder3, unet, vaeDecoder, vaeEncoder, controlnet, config.Diffusers, config.Schedulers, config.SchedulerOptions, logger);
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
        public static new StableDiffusion3Pipeline CreatePipeline(OnnxExecutionProvider executionProvider, string modelFolder, ModelType modelType = ModelType.Base, ILogger logger = default)
        {
            return CreatePipeline(ModelFactory.CreateStableDiffusion3ModelSet(modelFolder, modelType).WithProvider(executionProvider), logger);
        }
    }
}
