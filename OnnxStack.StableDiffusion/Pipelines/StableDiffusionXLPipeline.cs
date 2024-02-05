using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Diffusers.StableDiffusionXL;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Pipelines
{
    public class StableDiffusionXLPipeline : StableDiffusionPipeline
    {
        protected TokenizerModel _tokenizer2;
        protected OnnxModelSession _textEncoder2;

        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionXLPipeline"/> class.
        /// </summary>
        /// <param name="pipelineOptions">The pipeline options</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="tokenizer2">The tokenizer2.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="textEncoder2">The text encoder2.</param>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public StableDiffusionXLPipeline(PipelineOptions pipelineOptions, TokenizerModel tokenizer, TokenizerModel tokenizer2, TextEncoderModel textEncoder, TextEncoderModel textEncoder2, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, List<DiffuserType> diffusers, SchedulerOptions defaultSchedulerOptions = default, ILogger logger = default)
            : base(pipelineOptions, tokenizer, textEncoder, unet, vaeDecoder, vaeEncoder, diffusers, defaultSchedulerOptions, logger)
        {
            _tokenizer2 = tokenizer2;
            _textEncoder2 = textEncoder2;
            _supportedSchedulers = new List<SchedulerType>
            {
                SchedulerType.Euler,
                SchedulerType.EulerAncestral,
                SchedulerType.DDPM,
                SchedulerType.KDPM2
            };
            _defaultSchedulerOptions = defaultSchedulerOptions ?? new SchedulerOptions
            {
                Width = 1024,
                Height = 1024,
                InferenceSteps = 20,
                GuidanceScale = 5f,
                SchedulerType = SchedulerType.EulerAncestral
            };
        }


        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override DiffuserPipelineType PipelineType => DiffuserPipelineType.StableDiffusionXL;


        /// <summary>
        /// Loads the pipeline
        /// </summary>
        public override Task LoadAsync()
        {
            if (_pipelineOptions.MemoryMode == MemoryModeType.Minimum)
                return base.LoadAsync();

            // Preload all models into VRAM
            return Task.WhenAll
            (
                _tokenizer2.LoadAsync(),
                _textEncoder2.LoadAsync(),
                base.LoadAsync()
            );
        }


        /// <summary>
        /// Unloads the pipeline.
        /// </summary>
        /// <returns></returns>
        public override Task UnloadAsync()
        {
            _tokenizer2?.Dispose();
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
                DiffuserType.TextToImage => new TextDiffuser(_unet, _vaeDecoder, _vaeEncoder, _pipelineOptions.MemoryMode, _logger),
                DiffuserType.ImageToImage => new ImageDiffuser(_unet, _vaeDecoder, _vaeEncoder, _pipelineOptions.MemoryMode, _logger),
                DiffuserType.ImageInpaintLegacy => new InpaintLegacyDiffuser(_unet, _vaeDecoder, _vaeEncoder, _pipelineOptions.MemoryMode, _logger),
                DiffuserType.ControlNet => new ControlNetDiffuser(controlNetModel, _unet, _vaeDecoder, _vaeEncoder, _pipelineOptions.MemoryMode, _logger),
                DiffuserType.ControlNetImage => new ControlNetImageDiffuser(controlNetModel, _unet, _vaeDecoder, _vaeEncoder, _pipelineOptions.MemoryMode, _logger),
                _ => throw new NotImplementedException()
            };
        }


        /// <summary>
        /// Creates the prompt embeds.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="isGuidanceEnabled">if set to <c>true</c> [is guidance enabled].</param>
        /// <returns></returns>
        protected override async Task<PromptEmbeddingsResult> CreatePromptEmbedsAsync(PromptOptions promptOptions, bool isGuidanceEnabled)
        {
            return _unet.ModelType switch
            {
                ModelType.Refiner => await CreateEmbedsTwoAsync(promptOptions, isGuidanceEnabled),
                _ => await CreateEmbedsBothAsync(promptOptions, isGuidanceEnabled),
            };
        }


        /// <summary>
        /// Creates the embeds using Tokenizer2 and TextEncoder2
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="isGuidanceEnabled">if set to <c>true</c> [is guidance enabled].</param>
        /// <returns></returns>
        private async Task<PromptEmbeddingsResult> CreateEmbedsTwoAsync(PromptOptions promptOptions, bool isGuidanceEnabled)
        {
            /// Tokenize Prompt and NegativePrompt with Tokenizer2
            var promptTokens = await DecodeTextAsLongAsync(promptOptions.Prompt);
            var negativePromptTokens = await DecodeTextAsLongAsync(promptOptions.NegativePrompt);
            var maxPromptTokenCount = Math.Max(promptTokens.Length, negativePromptTokens.Length);

            // Generate embeds for tokens
            var promptEmbeddings = await GenerateEmbedsAsync(promptTokens, maxPromptTokenCount);
            var negativePromptEmbeddings = await GenerateEmbedsAsync(negativePromptTokens, maxPromptTokenCount);

            // Unload if required
            if (_pipelineOptions.MemoryMode == MemoryModeType.Minimum)
            {
                await _tokenizer2.UnloadAsync();
                await _textEncoder2.UnloadAsync();
            }

            if (isGuidanceEnabled)
                return new PromptEmbeddingsResult(
                    negativePromptEmbeddings.PromptEmbeds.Concatenate(promptEmbeddings.PromptEmbeds),
                    negativePromptEmbeddings.PooledPromptEmbeds.Concatenate(promptEmbeddings.PooledPromptEmbeds));

            return new PromptEmbeddingsResult(promptEmbeddings.PromptEmbeds, promptEmbeddings.PooledPromptEmbeds);
        }


        /// <summary>
        /// Creates the embeds using Tokenizer, Tokenizer2, TextEncoder and TextEncoder2
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="isGuidanceEnabled">if set to <c>true</c> is guidance enabled.</param>
        /// <returns></returns>
        private async Task<PromptEmbeddingsResult> CreateEmbedsBothAsync(PromptOptions promptOptions, bool isGuidanceEnabled)
        {
            // Tokenize Prompt and NegativePrompt
            var promptTokens = await DecodePromptTextAsync(promptOptions.Prompt);
            var negativePromptTokens = await DecodePromptTextAsync(promptOptions.NegativePrompt);
            var maxPromptTokenCount = Math.Max(promptTokens.Length, negativePromptTokens.Length);

            // Generate embeds for tokens
            var promptEmbeddings = await GeneratePromptEmbedsAsync(promptTokens, maxPromptTokenCount);
            var negativePromptEmbeddings = await GeneratePromptEmbedsAsync(negativePromptTokens, maxPromptTokenCount);

            /// Tokenize Prompt and NegativePrompt with Tokenizer2
            var dualPromptTokens = await DecodeTextAsLongAsync(promptOptions.Prompt);
            var dualNegativePromptTokens = await DecodeTextAsLongAsync(promptOptions.NegativePrompt);

            // Generate embeds for tokens
            var dualPromptEmbeddings = await GenerateEmbedsAsync(dualPromptTokens, maxPromptTokenCount);
            var dualNegativePromptEmbeddings = await GenerateEmbedsAsync(dualNegativePromptTokens, maxPromptTokenCount);

            var dualPrompt = promptEmbeddings.Concatenate(dualPromptEmbeddings.PromptEmbeds, 2);
            var dualNegativePrompt = negativePromptEmbeddings.Concatenate(dualNegativePromptEmbeddings.PromptEmbeds, 2);
            var pooledPromptEmbeds = dualPromptEmbeddings.PooledPromptEmbeds;
            var pooledNegativePromptEmbeds = dualNegativePromptEmbeddings.PooledPromptEmbeds;

            // Unload if required
            if (_pipelineOptions.MemoryMode == MemoryModeType.Minimum)
            {
                await _tokenizer2.UnloadAsync();
                await _textEncoder2.UnloadAsync();
            }

            if (isGuidanceEnabled)
                return new PromptEmbeddingsResult(dualNegativePrompt.Concatenate(dualPrompt), pooledNegativePromptEmbeds.Concatenate(pooledPromptEmbeds));

            return new PromptEmbeddingsResult(dualPrompt, pooledPromptEmbeds);
        }


        /// <summary>
        /// Decodes the text as tokens
        /// </summary>
        /// <param name="inputText">The input text.</param>
        /// <returns></returns>
        private async Task<long[]> DecodeTextAsLongAsync(string inputText)
        {
            if (string.IsNullOrEmpty(inputText))
                return Array.Empty<long>();

            var metadata = await _tokenizer2.GetMetadataAsync();
            var inputTensor = new DenseTensor<string>(new string[] { inputText }, new int[] { 1 });
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(inputTensor);
                inferenceParameters.AddOutputBuffer();

                using (var results = _tokenizer2.RunInference(inferenceParameters))
                {
                    var resultData = results.First().ToArray<long>();
                    return resultData;
                }
            }
        }


        /// <summary>
        /// Encodes the tokens.
        /// </summary>
        /// <param name="tokenizedInput">The tokenized input.</param>
        /// <returns></returns>
        private async Task<EncoderResult> EncodeTokensAsync(long[] tokenizedInput)
        {
            var inputDim = new[] { 1, tokenizedInput.Length };
            var promptOutputDim = new[] { 1, tokenizedInput.Length, _tokenizer2.TokenizerLength };
            var pooledOutputDim = new[] { 1, _tokenizer2.TokenizerLength };
            var metadata = await _textEncoder2.GetMetadataAsync();
            var inputTensor = new DenseTensor<long>(tokenizedInput, inputDim);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                int hiddenStateIndex = metadata.Outputs.Count - 2;
                inferenceParameters.AddInputTensor(inputTensor);
                inferenceParameters.AddOutputBuffer(pooledOutputDim);
                inferenceParameters.AddOutputBuffer(hiddenStateIndex, promptOutputDim);

                var results = await _textEncoder2.RunInferenceAsync(inferenceParameters);
                return new EncoderResult(results.Last().ToArray(), results.First().ToArray());
            }
        }


        /// <summary>
        /// Generates the embeds.
        /// </summary>
        /// <param name="inputTokens">The input tokens.</param>
        /// <param name="minimumLength">The minimum length.</param>
        /// <returns></returns>
        private async Task<EmbedsResult> GenerateEmbedsAsync(long[] inputTokens, int minimumLength)
        {
            // If less than minimumLength pad with blank tokens
            if (inputTokens.Length < minimumLength)
                inputTokens = PadWithBlankTokens(inputTokens, minimumLength).ToArray();

            // The CLIP tokenizer only supports 77 tokens, batch process in groups of 77 and concatenate1
            var embeddings = new List<float>();
            var pooledEmbeds = new List<float>();
            foreach (var tokenBatch in inputTokens.Batch(_tokenizer2.TokenizerLimit))
            {
                var tokens = PadWithBlankTokens(tokenBatch, _tokenizer2.TokenizerLimit);
                var result = await EncodeTokensAsync(tokens.ToArray());

                embeddings.AddRange(result.PromptEmbeds);
                pooledEmbeds.AddRange(result.PooledPromptEmbeds);
            }

            var embeddingsDim = new[] { 1, embeddings.Count / _tokenizer2.TokenizerLength, _tokenizer2.TokenizerLength };
            var promptTensor = TensorHelper.CreateTensor(embeddings.ToArray(), embeddingsDim);

            //TODO: Pooled embeds do not support more than 77 tokens, just grab first set
            var pooledDim = new[] { 1, _tokenizer2.TokenizerLength };
            var pooledTensor = TensorHelper.CreateTensor(pooledEmbeds.Take(_tokenizer2.TokenizerLength).ToArray(), pooledDim);
            return new EmbedsResult(promptTensor, pooledTensor);
        }


        /// <summary>
        /// Pads the input array with blank tokens.
        /// </summary>
        /// <param name="inputs">The inputs.</param>
        /// <param name="requiredLength">Length of the required.</param>
        /// <returns></returns>
        private IEnumerable<long> PadWithBlankTokens(IEnumerable<long> inputs, int requiredLength)
        {
            var count = inputs.Count();
            if (requiredLength > count)
                return inputs.Concat(Enumerable.Repeat((long)_tokenizer.PadTokenId, requiredLength - count));
            return inputs;
        }


        /// <summary>
        /// Creates the pipeline from a ModelSet configuration.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static new StableDiffusionXLPipeline CreatePipeline(StableDiffusionModelSet modelSet, ILogger logger = default)
        {
            var unet = new UNetConditionModel(modelSet.UnetConfig.ApplyDefaults(modelSet));
            var tokenizer = new TokenizerModel(modelSet.TokenizerConfig.ApplyDefaults(modelSet));
            var tokenizer2 = new TokenizerModel(modelSet.Tokenizer2Config.ApplyDefaults(modelSet));
            var textEncoder = new TextEncoderModel(modelSet.TextEncoderConfig.ApplyDefaults(modelSet));
            var textEncoder2 = new TextEncoderModel(modelSet.TextEncoder2Config.ApplyDefaults(modelSet));
            var vaeDecoder = new AutoEncoderModel(modelSet.VaeDecoderConfig.ApplyDefaults(modelSet));
            var vaeEncoder = new AutoEncoderModel(modelSet.VaeEncoderConfig.ApplyDefaults(modelSet));
            var pipelineOptions = new PipelineOptions(modelSet.Name, modelSet.MemoryMode);
            return new StableDiffusionXLPipeline(pipelineOptions, tokenizer, tokenizer2, textEncoder, textEncoder2, unet, vaeDecoder, vaeEncoder, modelSet.Diffusers, modelSet.SchedulerOptions, logger);
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
        public static new StableDiffusionXLPipeline CreatePipeline(string modelFolder, ModelType modelType = ModelType.Base, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML, MemoryModeType memoryMode = MemoryModeType.Maximum, ILogger logger = default)
        {
            return CreatePipeline(ModelFactory.CreateModelSet(modelFolder, DiffuserPipelineType.StableDiffusionXL, modelType, deviceId, executionProvider, memoryMode), logger);
        }
    }
}
