using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Diffusers.StableCascade;
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
    public sealed class StableCascadePipeline : StableDiffusionPipeline
    {
        private readonly UNetConditionModel _decoderUnet;

        /// <summary>
        /// Initializes a new instance of the <see cref="StableCascadePipeline"/> class.
        /// </summary>
        /// <param name="name">The pipeline name.</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="priorUnet">The prior unet.</param>
        /// <param name="decoderUnet">The decoder unet.</param>
        /// <param name="imageDecoder">The image decoder (VQGAN).</param>
        /// <param name="imageEncoder">The image encoder.</param>
        /// <param name="diffusers">The diffusers.</param>
        /// <param name="defaultSchedulerOptions">The default scheduler options.</param>
        /// <param name="logger">The logger.</param>
        public StableCascadePipeline(string name, ITokenizer tokenizer, TextEncoderModel textEncoder, UNetConditionModel priorUnet, UNetConditionModel decoderUnet, AutoEncoderModel imageDecoder, AutoEncoderModel imageEncoder, UNetConditionModel controlNet, List<DiffuserType> diffusers, List<SchedulerType> schedulers, SchedulerOptions defaultSchedulerOptions = default, ILogger logger = default)
            : base(name, tokenizer, textEncoder, priorUnet, imageDecoder, imageEncoder, controlNet, diffusers, schedulers, defaultSchedulerOptions, logger)
        {
            _decoderUnet = decoderUnet;
            _supportedDiffusers = diffusers ?? new List<DiffuserType>
            {
                DiffuserType.TextToImage,
                DiffuserType.ImageToImage
            };
            _supportedSchedulers = schedulers ?? new List<SchedulerType>
            {
                SchedulerType.DDPMWuerstchen
            };
            _defaultSchedulerOptions = defaultSchedulerOptions ?? new SchedulerOptions
            {
                Width = 1024,
                Height = 1024,
                InferenceSteps = 20,
                GuidanceScale = 4f,
                GuidanceScale2 = 0f,
                InferenceSteps2 = 10,
                SchedulerType = SchedulerType.DDPMWuerstchen
            };
        }


        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override PipelineType PipelineType => PipelineType.StableCascade;

        /// <summary>
        /// Gets the unet.
        /// </summary>
        public UNetConditionModel PriorUnet => _unet;

        /// <summary>
        /// Gets the unet.
        /// </summary>
        public UNetConditionModel DecoderUnet => _decoderUnet;


        /// <summary>
        /// Unloads the pipeline.
        /// </summary>
        /// <returns></returns>
        public override Task UnloadAsync()
        {
            _decoderUnet?.Dispose();
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
                DiffuserType.TextToImage => new TextDiffuser(_unet, _decoderUnet, _vaeDecoder, _logger),
                DiffuserType.ImageToImage => new ImageDiffuser(_unet, _decoderUnet, _vaeDecoder, _vaeEncoder, _logger),
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
            if (options.IsLowMemoryComputeEnabled && _decoderUnet?.Session is not null)
                await _decoderUnet.UnloadAsync();
        }


        /// <summary>
        /// Creates the embeds using Tokenizer2 and TextEncoder2
        /// </summary>
        /// <param name="options">The prompt options.</param>
        /// <returns></returns>
        protected override async Task<PromptEmbeddingsResult> CreatePromptEmbedsAsync(GenerateOptions options, CancellationToken cancellationToken = default)
        {
            /// Tokenize Prompt and NegativePrompt with Tokenizer2
            var timestamp = _logger?.LogBegin();
            var promptTokens = await DecodeTextAsLongAsync(options.Prompt, cancellationToken);
            var negativePromptTokens = await DecodeTextAsLongAsync(options.NegativePrompt, cancellationToken);
            var maxPromptTokenCount = Math.Max(promptTokens.InputIds.Length, negativePromptTokens.InputIds.Length);
            _logger?.LogEnd(LogLevel.Debug, $"Tokenizer", timestamp);


            // Generate embeds for tokens
            timestamp = _logger?.LogBegin();
            var promptEmbeddings = await GenerateEmbedsAsync(promptTokens, maxPromptTokenCount, cancellationToken);
            var negativePromptEmbeddings = await GenerateEmbedsAsync(negativePromptTokens, maxPromptTokenCount, cancellationToken);
            _logger?.LogEnd(LogLevel.Debug, $"TextEncoder", timestamp);

            // Unload if required
            if (options.IsLowMemoryTextEncoderEnabled)
            {
                await _textEncoder.UnloadAsync();
            }

            return new PromptEmbeddingsResult(promptEmbeddings.PromptEmbeds, promptEmbeddings.PooledPromptEmbeds, negativePromptEmbeddings.PromptEmbeds, negativePromptEmbeddings.PooledPromptEmbeds);
        }


        /// <summary>
        /// Decodes the text as tokens
        /// </summary>
        /// <param name="inputText">The input text.</param>
        /// <returns></returns>
        private async Task<TokenizerResult> DecodeTextAsLongAsync(string inputText, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(inputText))
                return new TokenizerResult(Array.Empty<long>(), Array.Empty<long>());

            return await _tokenizer.EncodeAsync(inputText);
        }


        /// <summary>
        /// Encodes the tokens.
        /// </summary>
        /// <param name="tokenizedInput">The tokenized input.</param>
        /// <returns></returns>
        private async Task<EncoderResult> EncodeTokensAsync(TokenizerResult tokenizedInput, CancellationToken cancellationToken = default)
        {
            var metadata = await _textEncoder.LoadAsync(cancellationToken: cancellationToken);
            var inputTensor = new DenseTensor<long>(tokenizedInput.InputIds, new[] { 1, tokenizedInput.InputIds.Length });
            var attentionTensor = new DenseTensor<long>(tokenizedInput.AttentionMask, new[] { 1, tokenizedInput.AttentionMask.Length });
            using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
            {
                inferenceParameters.AddInputTensor(inputTensor);
                inferenceParameters.AddInputTensor(attentionTensor);

                // text_embeds + hidden_states.32
                inferenceParameters.AddOutputBuffer(new[] { 1, _tokenizer.TokenizerLength });
                inferenceParameters.AddOutputBuffer(metadata.Outputs.Count - 1, new[] { 1, tokenizedInput.InputIds.Length, _tokenizer.TokenizerLength });

                var results = await _textEncoder.RunInferenceAsync(inferenceParameters);
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
        /// <param name="inputTokens">The input tokens.</param>
        /// <param name="minimumLength">The minimum length.</param>
        /// <returns></returns>
        private async Task<EncoderResult> GenerateEmbedsAsync(TokenizerResult inputTokens, int minimumLength, CancellationToken cancellationToken = default)
        {
            // If less than minimumLength pad with blank tokens
            if (inputTokens.InputIds.Length < minimumLength)
            {
                inputTokens.InputIds = PadWithBlankTokens(inputTokens.InputIds, minimumLength, _tokenizer.PadTokenId).ToArray();
                inputTokens.AttentionMask = PadWithBlankTokens(inputTokens.AttentionMask, minimumLength, 0).ToArray();
            }

            // The CLIP tokenizer only supports 77 tokens, batch process in groups of 77 and concatenate
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
                var result = await EncodeTokensAsync(new TokenizerResult(tokenBatches[i], attentionBatches[i]), cancellationToken);
                promptEmbeddings.AddRange(result.PromptEmbeds);
                pooledPromptEmbeddings.AddRange(result.PooledPromptEmbeds);
            }

            var promptTensor = new DenseTensor<float>(promptEmbeddings.ToArray(), new[] { 1, promptEmbeddings.Count / _tokenizer.TokenizerLength, _tokenizer.TokenizerLength });
            var pooledTensor = new DenseTensor<float>(pooledPromptEmbeddings.ToArray(), new[] { 1, tokenBatches.Count, _tokenizer.TokenizerLength });
            return new EncoderResult(promptTensor, pooledTensor);
        }


        /// <summary>
        /// Creates the pipeline from a ModelSet configuration.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static new StableCascadePipeline CreatePipeline(StableDiffusionModelSet modelSet, ILogger logger = default)
        {
            var config = modelSet with { };
            var priorUnet = new UNetConditionModel(config.UnetConfig.ApplyDefaults(config));
            var decoderUnet = new UNetConditionModel(config.Unet2Config.ApplyDefaults(config));
            var tokenizer = new ClipTokenizer(config.TokenizerConfig.ApplyDefaults(config));
            var textEncoder = new TextEncoderModel(config.TextEncoderConfig.ApplyDefaults(config));
            var imageDecoder = new AutoEncoderModel(config.VaeDecoderConfig.ApplyDefaults(config));
            var imageEncoder = new AutoEncoderModel(config.VaeEncoderConfig.ApplyDefaults(config));
            var controlnet = default(UNetConditionModel);
            if (config.ControlNetUnetConfig is not null)
                controlnet = new UNetConditionModel(config.ControlNetUnetConfig.ApplyDefaults(config));

            LogPipelineInfo(modelSet, logger);
            return new StableCascadePipeline(config.Name, tokenizer, textEncoder, priorUnet, decoderUnet, imageDecoder, imageEncoder, controlnet, config.Diffusers, config.Schedulers, config.SchedulerOptions, logger);
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
        public static new StableCascadePipeline CreatePipeline(OnnxExecutionProvider executionProvider, string modelFolder, ModelType modelType = ModelType.Base, ILogger logger = default)
        {
            return CreatePipeline(ModelFactory.CreateStableCascadeModelSet(modelFolder, modelType).WithProvider(executionProvider), logger);
        }
    }
}
