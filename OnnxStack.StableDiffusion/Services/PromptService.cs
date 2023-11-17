using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
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


        /// <summary>
        /// Creates the prompt & negative prompt embeddings.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="negativePrompt">The negative prompt.</param>
        /// <returns>Tensor containing all text embeds generated from the prompt and negative prompt</returns>
        public async Task<DenseTensor<float>> CreatePromptAsync(IModelOptions model, PromptOptions promptOptions, bool isGuidanceEnabled)
        {
            // Tokenize Prompt and NegativePrompt
            var promptTokens = await DecodeTextAsync(model, promptOptions.Prompt);
            var negativePromptTokens = await DecodeTextAsync(model, promptOptions.NegativePrompt);
            var maxPromptTokenCount = Math.Max(promptTokens.Length, negativePromptTokens.Length);

            // Generate embeds for tokens
            var promptEmbeddings = await GenerateEmbedsAsync(model, promptTokens, maxPromptTokenCount);
            var negativePromptEmbeddings = await GenerateEmbedsAsync(model, negativePromptTokens, maxPromptTokenCount);

            // If we are doing guided diffusion, concatenate the negative prompt embeddings
            // If not we ingore the negative prompt embeddings
            if (isGuidanceEnabled)
                return negativePromptEmbeddings.Concatenate(promptEmbeddings);

            return promptEmbeddings;
        }


        /// <summary>
        /// Tokenizes the input string
        /// </summary>
        /// <param name="inputText">The input text.</param>
        /// <returns>Tokens generated for the specified text input</returns>
        public Task<int[]> DecodeTextAsync(IModelOptions model, string inputText)
        {
            if (string.IsNullOrEmpty(inputText))
                return Task.FromResult(Array.Empty<int>());

            var metadata = _onnxModelService.GetModelMetadata(model, OnnxModelType.Tokenizer);
            var inputMetadata = metadata.Inputs[0];
            var outputMetadata = metadata.Outputs[0];

            var inputTensor = new DenseTensor<string>(new string[] { inputText }, new int[] { 1 });
            using (var inferenceParameters = new OnnxInferenceParameters())
            {
                inferenceParameters.AddInput(inputMetadata, OrtValue.CreateFromStringTensor(inputTensor));
                inferenceParameters.AddOutput(outputMetadata);

                using (var results = _onnxModelService.RunInference(model, OnnxModelType.Tokenizer, inferenceParameters))
                {
                    var resultData = results.First().GetTensorDataAsSpan<long>().ToArray();
                    return Task.FromResult(Array.ConvertAll(resultData, Convert.ToInt32));
                }
            }
        }


        /// <summary>
        /// Encodes the tokens.
        /// </summary>
        /// <param name="tokenizedInput">The tokenized input.</param>
        /// <returns></returns>
        public async Task<float[]> EncodeTokensAsync(IModelOptions model, int[] tokenizedInput)
        {
            var metadata = _onnxModelService.GetModelMetadata(model, OnnxModelType.TextEncoder);
            var inputMetadata = metadata.Inputs[0];
            var outputMetadata = metadata.Outputs[0];

            var inputDim = new[] { 1L, tokenizedInput.Length };
            var outputDim = new[] { 1L, tokenizedInput.Length, model.EmbeddingsLength };
            using (var inferenceParameters = new OnnxInferenceParameters())
            {
                inferenceParameters.AddInput(inputMetadata, OrtValue.CreateTensorValueFromMemory(tokenizedInput, inputDim));
                inferenceParameters.AddOutput(outputMetadata, outputMetadata.CreateOutputBuffer(outputDim.ToInt()));

                var results = await _onnxModelService.RunInferenceAsync(model, OnnxModelType.TextEncoder, inferenceParameters);
                using (var result = results.First())
                {
                    return result.ToArray();
                }
            }
        }


        /// <summary>
        /// Generates the embeds for the input tokens.
        /// </summary>
        /// <param name="inputTokens">The input tokens.</param>
        /// <param name="minimumLength">The minimum length.</param>
        /// <returns></returns>
        private async Task<DenseTensor<float>> GenerateEmbedsAsync(IModelOptions model, int[] inputTokens, int minimumLength)
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

            var dim = new[] { 1, embeddings.Count / model.EmbeddingsLength, model.EmbeddingsLength };
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
        /// Helper for creating the input parameters.
        /// </summary>
        /// <param name="parameters">The parameters.</param>
        /// <returns></returns>
        private static IReadOnlyCollection<NamedOnnxValue> CreateInputParameters(params NamedOnnxValue[] parameters)
        {
            return parameters.ToList().AsReadOnly();
        }
    }
}
