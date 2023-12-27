using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Services
{

    public sealed class PromptService : IPromptService
    {
        private readonly IOnnxModelService _onnxModelService;

        /// <summary>
        /// Initializes a new instance of the <see cref="PromptService"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="onnxModelService">The onnx model service.</param>
        public PromptService(IOnnxModelService onnxModelService)
        {
            _onnxModelService = onnxModelService;
        }

        public record EncoderResult(float[] PromptEmbeds, float[] PooledPromptEmbeds);
        public record EmbedsResult(DenseTensor<float> PromptEmbeds, DenseTensor<float> PooledPromptEmbeds);

        /// <summary>
        /// Creates the prompt & negative prompt embeddings.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="negativePrompt">The negative prompt.</param>
        /// <returns>Tensor containing all text embeds generated from the prompt and negative prompt</returns>
        public async Task<PromptEmbeddingsResult> CreatePromptAsync(StableDiffusionModelSet model, PromptOptions promptOptions, bool isGuidanceEnabled)
        {
            return model.TokenizerType switch
            {
                TokenizerType.One => await CreateEmbedsOneAsync(model, promptOptions, isGuidanceEnabled),
                TokenizerType.Two => await CreateEmbedsTwoAsync(model, promptOptions, isGuidanceEnabled),
                TokenizerType.Both => await CreateEmbedsBothAsync(model, promptOptions, isGuidanceEnabled),
                _ => throw new ArgumentException("TokenizerType is not set")
            };
        }


        /// <summary>
        /// Creates the embeds using Tokenizer and TextEncoder
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="isGuidanceEnabled">if set to <c>true</c> is guidance enabled.</param>
        /// <returns></returns>
        private async Task<PromptEmbeddingsResult> CreateEmbedsOneAsync(StableDiffusionModelSet model, PromptOptions promptOptions, bool isGuidanceEnabled)
        {
            // Tokenize Prompt and NegativePrompt
            var promptTokens = await DecodeTextAsIntAsync(model, promptOptions.Prompt);
            var negativePromptTokens = await DecodeTextAsIntAsync(model, promptOptions.NegativePrompt);
            var maxPromptTokenCount = Math.Max(promptTokens.Length, negativePromptTokens.Length);

            // Generate embeds for tokens
            var promptEmbeddings = await GenerateEmbedsAsync(model, promptTokens, maxPromptTokenCount);
            var negativePromptEmbeddings = await GenerateEmbedsAsync(model, negativePromptTokens, maxPromptTokenCount);

            if (isGuidanceEnabled)
                return new PromptEmbeddingsResult(negativePromptEmbeddings.Concatenate(promptEmbeddings));

            return new PromptEmbeddingsResult(promptEmbeddings);
        }

        /// <summary>
        /// Creates the embeds using Tokenizer2 and TextEncoder2
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="isGuidanceEnabled">if set to <c>true</c> is guidance enabled.</param>
        /// <returns></returns>
        private async Task<PromptEmbeddingsResult> CreateEmbedsTwoAsync(StableDiffusionModelSet model, PromptOptions promptOptions, bool isGuidanceEnabled)
        {
            /// Tokenize Prompt and NegativePrompt with Tokenizer2
            var promptTokens = await DecodeTextAsLongAsync(model, promptOptions.Prompt);
            var negativePromptTokens = await DecodeTextAsLongAsync(model, promptOptions.NegativePrompt);
            var maxPromptTokenCount = Math.Max(promptTokens.Length, negativePromptTokens.Length);

            // Generate embeds for tokens
            var promptEmbeddings = await GenerateEmbedsAsync(model, promptTokens, maxPromptTokenCount);
            var negativePromptEmbeddings = await GenerateEmbedsAsync(model, negativePromptTokens, maxPromptTokenCount);

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
        private async Task<PromptEmbeddingsResult> CreateEmbedsBothAsync(StableDiffusionModelSet model, PromptOptions promptOptions, bool isGuidanceEnabled)
        {
            // Tokenize Prompt and NegativePrompt
            var promptTokens = await DecodeTextAsIntAsync(model, promptOptions.Prompt);
            var negativePromptTokens = await DecodeTextAsIntAsync(model, promptOptions.NegativePrompt);
            var maxPromptTokenCount = Math.Max(promptTokens.Length, negativePromptTokens.Length);

            // Generate embeds for tokens
            var promptEmbeddings = await GenerateEmbedsAsync(model, promptTokens, maxPromptTokenCount);
            var negativePromptEmbeddings = await GenerateEmbedsAsync(model, negativePromptTokens, maxPromptTokenCount);

            /// Tokenize Prompt and NegativePrompt with Tokenizer2
            var dualPromptTokens = await DecodeTextAsLongAsync(model, promptOptions.Prompt);
            var dualNegativePromptTokens = await DecodeTextAsLongAsync(model, promptOptions.NegativePrompt);

            // Generate embeds for tokens
            var dualPromptEmbeddings = await GenerateEmbedsAsync(model, dualPromptTokens, maxPromptTokenCount);
            var dualNegativePromptEmbeddings = await GenerateEmbedsAsync(model, dualNegativePromptTokens, maxPromptTokenCount);

            var dualPrompt = promptEmbeddings.Concatenate(dualPromptEmbeddings.PromptEmbeds, 2);
            var dualNegativePrompt = negativePromptEmbeddings.Concatenate(dualNegativePromptEmbeddings.PromptEmbeds, 2);
            var pooledPromptEmbeds = dualPromptEmbeddings.PooledPromptEmbeds;
            var pooledNegativePromptEmbeds = dualNegativePromptEmbeddings.PooledPromptEmbeds;

            if (isGuidanceEnabled)
                return new PromptEmbeddingsResult(dualNegativePrompt.Concatenate(dualPrompt), pooledNegativePromptEmbeds.Concatenate(pooledPromptEmbeds));

            return new PromptEmbeddingsResult(dualPrompt, pooledPromptEmbeds);
        }


        /// <summary>
        /// Tokenizes the input string to an int array
        /// </summary>
        /// <param name="inputText">The input text.</param>
        /// <returns>Tokens generated for the specified text input</returns>
        private Task<int[]> DecodeTextAsIntAsync(StableDiffusionModelSet model, string inputText)
        {
            if (string.IsNullOrEmpty(inputText))
                return Task.FromResult(Array.Empty<int>());

            var metadata = _onnxModelService.GetModelMetadata(model, OnnxModelType.Tokenizer);
            var inputTensor = new DenseTensor<string>(new string[] { inputText }, new int[] { 1 });
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(inputTensor);
                inferenceParameters.AddOutputBuffer();

                using (var results = _onnxModelService.RunInference(model, OnnxModelType.Tokenizer, inferenceParameters))
                {
                    var resultData = results.First().GetTensorDataAsSpan<long>();
                    return Task.FromResult(Array.ConvertAll(resultData.ToArray(), Convert.ToInt32));
                }
            }
        }


        /// <summary>
        /// Tokenizes the input string to an long array
        /// </summary>
        /// <param name="inputText">The input text.</param>
        /// <returns>Tokens generated for the specified text input</returns>
        private Task<long[]> DecodeTextAsLongAsync(StableDiffusionModelSet model, string inputText)
        {
            if (string.IsNullOrEmpty(inputText))
                return Task.FromResult(Array.Empty<long>());

            var metadata = _onnxModelService.GetModelMetadata(model, OnnxModelType.Tokenizer2);
            var inputTensor = new DenseTensor<string>(new string[] { inputText }, new int[] { 1 });
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(inputTensor);
                inferenceParameters.AddOutputBuffer();

                using (var results = _onnxModelService.RunInference(model, OnnxModelType.Tokenizer2, inferenceParameters))
                {
                    var resultData = results.First().GetTensorDataAsSpan<long>();
                    return Task.FromResult(resultData.ToArray());
                }
            }
        }


        /// <summary>
        /// Encodes the tokens.
        /// </summary>
        /// <param name="tokenizedInput">The tokenized input.</param>
        /// <returns></returns>
        private async Task<float[]> EncodeTokensAsync(StableDiffusionModelSet model, int[] tokenizedInput)
        {
            var inputDim = new[] { 1, tokenizedInput.Length };
            var outputDim = new[] { 1, tokenizedInput.Length, model.TokenizerLength };
            var metadata = _onnxModelService.GetModelMetadata(model, OnnxModelType.TextEncoder);
            var inputTensor = new DenseTensor<int>(tokenizedInput, inputDim);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(inputTensor);
                inferenceParameters.AddOutputBuffer(outputDim);

                var results = await _onnxModelService.RunInferenceAsync(model, OnnxModelType.TextEncoder, inferenceParameters);
                using (var result = results.First())
                {
                    return result.ToArray();
                }
            }
        }


        /// <summary>
        /// Encodes the tokens.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="tokenizedInput">The tokenized input.</param>
        /// <returns></returns>
        private async Task<EncoderResult> EncodeTokensAsync(StableDiffusionModelSet model, long[] tokenizedInput)
        {
            var inputDim = new[] { 1, tokenizedInput.Length };
            var promptOutputDim = new[] { 1, tokenizedInput.Length, model.Tokenizer2Length };
            var pooledOutputDim = new[] { 1, model.Tokenizer2Length };
            var metadata = _onnxModelService.GetModelMetadata(model, OnnxModelType.TextEncoder2);
            var inputTensor = new DenseTensor<long>(tokenizedInput, inputDim);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                int hiddenStateIndex = metadata.Outputs.Count - 2;
                inferenceParameters.AddInputTensor(inputTensor);
                inferenceParameters.AddOutputBuffer(pooledOutputDim);
                inferenceParameters.AddOutputBuffer(hiddenStateIndex, promptOutputDim);

                var results = await _onnxModelService.RunInferenceAsync(model, OnnxModelType.TextEncoder2, inferenceParameters);
                return new EncoderResult(results.Last().ToArray(), results.First().ToArray());
            }
        }



        /// <summary>
        /// Generates the embeds for the input tokens.
        /// </summary>
        /// <param name="inputTokens">The input tokens.</param>
        /// <param name="minimumLength">The minimum length.</param>
        /// <returns></returns>
        private async Task<EmbedsResult> GenerateEmbedsAsync(StableDiffusionModelSet model, long[] inputTokens, int minimumLength)
        {
            // If less than minimumLength pad with blank tokens
            if (inputTokens.Length < minimumLength)
                inputTokens = PadWithBlankTokens(inputTokens, minimumLength, model.BlankTokenValueArray).ToArray();

            // The CLIP tokenizer only supports 77 tokens, batch process in groups of 77 and concatenate1
            var embeddings = new List<float>();
            var pooledEmbeds = new List<float>();
            foreach (var tokenBatch in inputTokens.Batch(model.TokenizerLimit))
            {
                var tokens = PadWithBlankTokens(tokenBatch, model.TokenizerLimit, model.BlankTokenValueArray);
                var result = await EncodeTokensAsync(model, tokens.ToArray());

                embeddings.AddRange(result.PromptEmbeds);
                pooledEmbeds.AddRange(result.PooledPromptEmbeds);
            }

            var embeddingsDim = new[] { 1, embeddings.Count / model.Tokenizer2Length, model.Tokenizer2Length };
            var promptTensor = TensorHelper.CreateTensor(embeddings.ToArray(), embeddingsDim);

            //TODO: Pooled embeds do not support more than 77 tokens, just grab first set
            var pooledDim = new[] { 1, model.Tokenizer2Length };
            var pooledTensor = TensorHelper.CreateTensor(pooledEmbeds.Take(model.Tokenizer2Length).ToArray(), pooledDim);
            return new EmbedsResult(promptTensor, pooledTensor);
        }


        /// <summary>
        /// Generates the embeds for the input tokens.
        /// </summary>
        /// <param name="inputTokens">The input tokens.</param>
        /// <param name="minimumLength">The minimum length.</param>
        /// <returns></returns>
        private async Task<DenseTensor<float>> GenerateEmbedsAsync(StableDiffusionModelSet model, int[] inputTokens, int minimumLength)
        {
            // If less than minimumLength pad with blank tokens
            if (inputTokens.Length < minimumLength)
                inputTokens = PadWithBlankTokens(inputTokens, minimumLength, model.BlankTokenValueArray).ToArray();

            // The CLIP tokenizer only supports 77 tokens, batch process in groups of 77 and concatenate
            var embeddings = new List<float>();
            foreach (var tokenBatch in inputTokens.Batch(model.TokenizerLimit))
            {
                var tokens = PadWithBlankTokens(tokenBatch, model.TokenizerLimit, model.BlankTokenValueArray);
                embeddings.AddRange(await EncodeTokensAsync(model, tokens.ToArray()));
            }

            var dim = new[] { 1, embeddings.Count / model.TokenizerLength, model.TokenizerLength };
            return TensorHelper.CreateTensor(embeddings.ToArray(), dim);
        }


        /// <summary>
        /// Pads a source sequence with blank tokens if its less that the required length.
        /// </summary>
        /// <param name="inputs">The inputs.</param>
        /// <param name="requiredLength">The the required length of the returned array.</param>
        /// <returns></returns>
        private IEnumerable<int> PadWithBlankTokens(IEnumerable<int> inputs, int requiredLength, ImmutableArray<int> blankTokens)
        {
            var count = inputs.Count();
            if (requiredLength > count)
                return inputs.Concat(blankTokens[..(requiredLength - count)]);
            return inputs;
        }


        /// <summary>
        /// Pads a source sequence with blank tokens if its less that the required length.
        /// </summary>
        /// <param name="inputs">The inputs.</param>
        /// <param name="requiredLength">The the required length of the returned array.</param>
        /// <param name="blankTokens">The blank tokens.</param>
        /// <returns></returns>
        private IEnumerable<long> PadWithBlankTokens(IEnumerable<long> inputs, int requiredLength, ImmutableArray<int> blankTokens)
        {
            var count = inputs.Count();
            if (requiredLength > count)
                return inputs.Concat(blankTokens[..(requiredLength - count)].ToArray().ToLong());
            return inputs;
        }
    }
}
