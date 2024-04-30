using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Diffusers.StableCascade;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Pipelines
{
    public sealed class StableCascadePipeline : StableDiffusionPipeline
    {
        private readonly UNetConditionModel _decoderUnet;

        /// <summary>
        /// Initializes a new instance of the <see cref="StableCascadePipeline"/> class.
        /// </summary>
        /// <param name="pipelineOptions">The pipeline options.</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="priorUnet">The prior unet.</param>
        /// <param name="decoderUnet">The decoder unet.</param>
        /// <param name="imageDecoder">The image decoder (VQGAN).</param>
        /// <param name="imageEncoder">The image encoder.</param>
        /// <param name="diffusers">The diffusers.</param>
        /// <param name="defaultSchedulerOptions">The default scheduler options.</param>
        /// <param name="logger">The logger.</param>
        public StableCascadePipeline(PipelineOptions pipelineOptions, TokenizerModel tokenizer, TextEncoderModel textEncoder, UNetConditionModel priorUnet, UNetConditionModel decoderUnet, AutoEncoderModel imageDecoder, AutoEncoderModel imageEncoder, List<DiffuserType> diffusers, SchedulerOptions defaultSchedulerOptions = default, ILogger logger = default)
            : base(pipelineOptions, tokenizer, textEncoder, priorUnet, imageDecoder, imageEncoder, diffusers, defaultSchedulerOptions, logger)
        {
            _decoderUnet = decoderUnet;
            _supportedDiffusers = diffusers ?? new List<DiffuserType>
            {
                DiffuserType.TextToImage,
                DiffuserType.ImageToImage
            };
            _supportedSchedulers = new List<SchedulerType>
            {
                SchedulerType.DDPM,
                SchedulerType.DDPMWuerstchen
            };
            _defaultSchedulerOptions = defaultSchedulerOptions ?? new SchedulerOptions
            {
                Width = 1024,
                Height = 1024,
                InferenceSteps = 20,
                GuidanceScale = 4f,
                SchedulerType = SchedulerType.DDPMWuerstchen
            };
        }


        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override DiffuserPipelineType PipelineType => DiffuserPipelineType.StableCascade;


        public override Task LoadAsync()
        {
            if (_pipelineOptions.MemoryMode == MemoryModeType.Minimum)
                return base.LoadAsync();

            // Preload all models into VRAM
            return Task.WhenAll
            (
                _decoderUnet.LoadAsync(),
                base.LoadAsync()
            );
        }


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
                DiffuserType.TextToImage => new TextDiffuser(_unet, _decoderUnet, _vaeDecoder, _pipelineOptions.MemoryMode, _logger),
                DiffuserType.ImageToImage => new ImageDiffuser(_unet, _decoderUnet, _vaeDecoder, _vaeEncoder, _pipelineOptions.MemoryMode, _logger),
                _ => throw new NotImplementedException()
            };
        }


        /// <summary>
        /// Check if we should run guidance.
        /// </summary>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <returns></returns>
        protected override bool ShouldPerformGuidance(SchedulerOptions schedulerOptions)
        {
            return schedulerOptions.GuidanceScale > 1f || schedulerOptions.GuidanceScale2 > 1f;
        }


        /// <summary>
        /// Creates the embeds using Tokenizer2 and TextEncoder2
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="isGuidanceEnabled">if set to <c>true</c> [is guidance enabled].</param>
        /// <returns></returns>
        protected override async Task<PromptEmbeddingsResult> CreatePromptEmbedsAsync(PromptOptions promptOptions, bool isGuidanceEnabled)
        {
            /// Tokenize Prompt and NegativePrompt with Tokenizer2
            var promptTokens = await DecodeTextAsLongAsync(promptOptions.Prompt);
            var negativePromptTokens = await DecodeTextAsLongAsync(promptOptions.NegativePrompt);
            var maxPromptTokenCount = Math.Max(promptTokens.InputIds.Length, negativePromptTokens.InputIds.Length);

            // Generate embeds for tokens
            var promptEmbeddings = await GenerateEmbedsAsync(promptTokens, maxPromptTokenCount);
            var negativePromptEmbeddings = await GenerateEmbedsAsync(negativePromptTokens, maxPromptTokenCount);

            // Unload if required
            if (_pipelineOptions.MemoryMode == MemoryModeType.Minimum)
            {
                await _tokenizer.UnloadAsync();
                await _textEncoder.UnloadAsync();
            }

            if (isGuidanceEnabled)
                return new PromptEmbeddingsResult(
                    negativePromptEmbeddings.PromptEmbeds.Concatenate(promptEmbeddings.PromptEmbeds),
                    negativePromptEmbeddings.PooledPromptEmbeds.Concatenate(promptEmbeddings.PooledPromptEmbeds));

            return new PromptEmbeddingsResult(promptEmbeddings.PromptEmbeds, promptEmbeddings.PooledPromptEmbeds);
        }


        /// <summary>
        /// Decodes the text as tokens
        /// </summary>
        /// <param name="inputText">The input text.</param>
        /// <returns></returns>
        private async Task<TokenizerResult> DecodeTextAsLongAsync(string inputText)
        {
            if (string.IsNullOrEmpty(inputText))
                return new TokenizerResult(Array.Empty<long>(), Array.Empty<long>());

            var metadata = await _tokenizer.GetMetadataAsync();
            var inputTensor = new DenseTensor<string>(new string[] { inputText }, new int[] { 1 });
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(inputTensor);
                inferenceParameters.AddOutputBuffer();
                inferenceParameters.AddOutputBuffer();

                using (var results = _tokenizer.RunInference(inferenceParameters))
                {
                    return new TokenizerResult(results[0].ToArray<long>(), results[1].ToArray<long>());
                }
            }
        }


        /// <summary>
        /// Encodes the tokens.
        /// </summary>
        /// <param name="tokenizedInput">The tokenized input.</param>
        /// <returns></returns>
        private async Task<EncoderResult> EncodeTokensAsync(TokenizerResult tokenizedInput)
        {
            var metadata = await _textEncoder.GetMetadataAsync();
            var inputTensor = new DenseTensor<long>(tokenizedInput.InputIds, new[] { 1, tokenizedInput.InputIds.Length });
            var attentionTensor = new DenseTensor<long>(tokenizedInput.AttentionMask, new[] { 1, tokenizedInput.AttentionMask.Length });
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(inputTensor);
                inferenceParameters.AddInputTensor(attentionTensor);

                // text_embeds + hidden_states.32
                inferenceParameters.AddOutputBuffer(new[] { 1, _tokenizer.TokenizerLength });
                inferenceParameters.AddOutputBuffer(metadata.Outputs.Count - 1, new[] { 1, tokenizedInput.InputIds.Length, _tokenizer.TokenizerLength });

                var results = await _textEncoder.RunInferenceAsync(inferenceParameters);
                var promptEmbeds = results.Last().ToDenseTensor();
                var promptEmbedsPooled = results.First().ToDenseTensor();
                return new EncoderResult(promptEmbeds, promptEmbedsPooled);
            }
        }


        /// <summary>
        /// Generates the embeds.
        /// </summary>
        /// <param name="inputTokens">The input tokens.</param>
        /// <param name="minimumLength">The minimum length.</param>
        /// <returns></returns>
        private async Task<PromptEmbeddingsResult> GenerateEmbedsAsync(TokenizerResult inputTokens, int minimumLength)
        {
            // If less than minimumLength pad with blank tokens
            if (inputTokens.InputIds.Length < minimumLength)
            {
                inputTokens.InputIds = PadWithBlankTokens(inputTokens.InputIds, minimumLength, _tokenizer.PadTokenId).ToArray();
                inputTokens.AttentionMask = PadWithBlankTokens(inputTokens.AttentionMask, minimumLength, 1).ToArray();
            }

            // The CLIP tokenizer only supports 77 tokens, batch process in groups of 77 and concatenate
            var tokenBatches = new List<long[]>();
            var attentionBatches = new List<long[]>();
            foreach (var tokenBatch in inputTokens.InputIds.Batch(_tokenizer.TokenizerLimit))
                tokenBatches.Add(PadWithBlankTokens(tokenBatch, _tokenizer.TokenizerLimit, _tokenizer.PadTokenId).ToArray());
            foreach (var attentionBatch in inputTokens.AttentionMask.Batch(_tokenizer.TokenizerLimit))
                attentionBatches.Add(PadWithBlankTokens(attentionBatch, _tokenizer.TokenizerLimit, 1).ToArray());

            var promptEmbeddings = new List<float>();
            var pooledPromptEmbeddings = new List<float>();
            for (int i = 0; i < tokenBatches.Count; i++)
            {
                var result = await EncodeTokensAsync(new TokenizerResult(tokenBatches[i], attentionBatches[i]));
                promptEmbeddings.AddRange(result.PromptEmbeds);
                pooledPromptEmbeddings.AddRange(result.PooledPromptEmbeds);
            }

            var promptTensor = new DenseTensor<float>(promptEmbeddings.ToArray(), new[] { 1, promptEmbeddings.Count / _tokenizer.TokenizerLength, _tokenizer.TokenizerLength });
            var pooledTensor = new DenseTensor<float>(pooledPromptEmbeddings.ToArray(), new[] { 1, tokenBatches.Count, _tokenizer.TokenizerLength });
            return new PromptEmbeddingsResult(promptTensor, pooledTensor);
        }


        /// <summary>
        /// Creates the pipeline from a ModelSet configuration.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static new StableCascadePipeline CreatePipeline(StableDiffusionModelSet modelSet, ILogger logger = default)
        {
            var priorUnet = new UNetConditionModel(modelSet.UnetConfig.ApplyDefaults(modelSet));
            var decoderUnet = new UNetConditionModel(modelSet.DecoderUnetConfig.ApplyDefaults(modelSet));
            var tokenizer = new TokenizerModel(modelSet.TokenizerConfig.ApplyDefaults(modelSet));
            var textEncoder = new TextEncoderModel(modelSet.TextEncoderConfig.ApplyDefaults(modelSet));
            var imageDecoder = new AutoEncoderModel(modelSet.VaeDecoderConfig.ApplyDefaults(modelSet));
            var imageEncoder = new AutoEncoderModel(modelSet.VaeEncoderConfig.ApplyDefaults(modelSet));
            var pipelineOptions = new PipelineOptions(modelSet.Name, modelSet.MemoryMode);
            return new StableCascadePipeline(pipelineOptions, tokenizer, textEncoder, priorUnet, decoderUnet, imageDecoder, imageEncoder, modelSet.Diffusers, modelSet.SchedulerOptions, logger);
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
        public static new StableCascadePipeline CreatePipeline(string modelFolder, ModelType modelType = ModelType.Base, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML, MemoryModeType memoryMode = MemoryModeType.Maximum, ILogger logger = default)
        {
            return CreatePipeline(ModelFactory.CreateModelSet(modelFolder, DiffuserPipelineType.StableCascade, modelType, deviceId, executionProvider, memoryMode), logger);
        }
    }
}
