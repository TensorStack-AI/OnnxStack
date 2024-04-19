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
                DiffuserType.TextToImage
            };
            _supportedSchedulers = new List<SchedulerType>
            {
                SchedulerType.EulerAncestral
            };
            _defaultSchedulerOptions = defaultSchedulerOptions ?? new SchedulerOptions
            {
                InferenceSteps = 1,
                GuidanceScale = 0f,
                SchedulerType = SchedulerType.EulerAncestral
            };
        }


        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override DiffuserPipelineType PipelineType => DiffuserPipelineType.StableCascade;


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
                _ => throw new NotImplementedException()
            };
        }

        protected override async Task<PromptEmbeddingsResult> CreatePromptEmbedsAsync(PromptOptions promptOptions, bool isGuidanceEnabled)
        {
            /// Tokenize Prompt and NegativePrompt with Tokenizer2
            var promptTokens = await DecodePromptTextAsync(promptOptions.Prompt);
            var negativePromptTokens = await DecodePromptTextAsync(promptOptions.NegativePrompt);
            var maxPromptTokenCount = Math.Max(promptTokens.Length, negativePromptTokens.Length);

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


        private async Task<EncoderResult> EncodeTokensAsync(int[] tokenizedInput)
        {
            var inputDim = new[] { 1, tokenizedInput.Length };
            var promptOutputDim = new[] { 1, tokenizedInput.Length, _tokenizer.TokenizerLength };
            var pooledOutputDim = new[] { 1, _tokenizer.TokenizerLength };
            var metadata = await _textEncoder.GetMetadataAsync();
            var inputTensor = new DenseTensor<int>(tokenizedInput, inputDim);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(inputTensor);
                inferenceParameters.AddOutputBuffer(pooledOutputDim);
                inferenceParameters.AddOutputBuffer(promptOutputDim);

                var results = await _textEncoder.RunInferenceAsync(inferenceParameters);
                return new EncoderResult(results.Last().ToArray(), results.First().ToArray());
            }
        }


        private async Task<EmbedsResult> GenerateEmbedsAsync(int[] inputTokens, int minimumLength)
        {
            // If less than minimumLength pad with blank tokens
            if (inputTokens.Length < minimumLength)
                inputTokens = PadWithBlankTokens(inputTokens, minimumLength).ToArray();

            // The CLIP tokenizer only supports 77 tokens, batch process in groups of 77 and concatenate1
            var embeddings = new List<float>();
            var pooledEmbeds = new List<float>();
            foreach (var tokenBatch in inputTokens.Batch(_tokenizer.TokenizerLimit))
            {
                var tokens = PadWithBlankTokens(tokenBatch, _tokenizer.TokenizerLimit);
                var result = await EncodeTokensAsync(tokens.ToArray());

                embeddings.AddRange(result.PromptEmbeds);
                pooledEmbeds.AddRange(result.PooledPromptEmbeds);
            }

            var embeddingsDim = new[] { 1, embeddings.Count / _tokenizer.TokenizerLength, _tokenizer.TokenizerLength };
            var promptTensor = new DenseTensor<float>(embeddings.ToArray(), embeddingsDim);

            //TODO: Pooled embeds do not support more than 77 tokens, just grab first set
            var pooledDim = new[] { 1, 1, _tokenizer.TokenizerLength };
            var pooledTensor = new DenseTensor<float>(pooledEmbeds.Take(_tokenizer.TokenizerLength).ToArray(), pooledDim);
            return new EmbedsResult(promptTensor, pooledTensor);
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
