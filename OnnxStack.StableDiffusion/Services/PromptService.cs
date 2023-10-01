using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;

namespace OnnxStack.StableDiffusion.Services
{
    public sealed class PromptService : IPromptService
    {
        private readonly OnnxStackConfig _configuration;
        private readonly IOnnxModelService _onnxModelService;

        /// <summary>
        /// Initializes a new instance of the <see cref="PromptService"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="onnxModelService">The onnx model service.</param>
        public PromptService(OnnxStackConfig configuration, IOnnxModelService onnxModelService)
        {
            _configuration = configuration;
            _onnxModelService = onnxModelService;
        }


        /// <summary>
        /// Creates the prompt & negative prompt embeddings.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="negativePrompt">The negative prompt.</param>
        /// <returns>Tensor containing all text embeds generated from the prompt and negative prompt</returns>
        public async Task<DenseTensor<float>> CreatePromptAsync(string prompt, string negativePrompt)
        {
            // Tokenize Prompt and NegativePrompt
            var promptTokens = await DecodeTextAsync(prompt);
            var negativePromptTokens = await DecodeTextAsync(negativePrompt);
            var maxPromptTokenCount = Math.Max(promptTokens.Length, negativePromptTokens.Length);

            Console.WriteLine($"Prompt -   Length: {prompt.Length}, Tokens: {promptTokens.Length}");
            Console.WriteLine($"N-Prompt - Length: {negativePrompt?.Length}, Tokens: {negativePromptTokens.Length}");

            // Generate embeds for tokens
            var promptEmbeddings = await GenerateEmbedsAsync(promptTokens, maxPromptTokenCount);
            var negativePromptEmbeddings = await GenerateEmbedsAsync(negativePromptTokens, maxPromptTokenCount);

            // Calculate embeddings
            var textEmbeddings = new DenseTensor<float>(new[] { 2, promptEmbeddings.Count / Constants.ClipTokenizerEmbeddingsLength, Constants.ClipTokenizerEmbeddingsLength });
            for (var i = 0; i < promptEmbeddings.Count; i++)
            {
                textEmbeddings[0, i / Constants.ClipTokenizerEmbeddingsLength, i % Constants.ClipTokenizerEmbeddingsLength] = negativePromptEmbeddings[i];
                textEmbeddings[1, i / Constants.ClipTokenizerEmbeddingsLength, i % Constants.ClipTokenizerEmbeddingsLength] = promptEmbeddings[i];
            }
            return textEmbeddings;
        }


        /// <summary>
        /// Tokenizes the input string
        /// </summary>
        /// <param name="inputText">The input text.</param>
        /// <returns>Tokens generated for the specified text input</returns>
        public async Task<int[]> DecodeTextAsync(string inputText)
        {
            if (string.IsNullOrEmpty(inputText))
                return Array.Empty<int>();

            // Create input tensor.
            var inputTensor = new DenseTensor<string>(new string[] { inputText }, new int[] { 1 });
            var inputParameters = CreateInputParameters(NamedOnnxValue.CreateFromTensor("string_input", inputTensor));

            // Run inference.
            using (var inferResult = await _onnxModelService.RunInferenceAsync(OnnxModelType.Tokenizer, inputParameters))
            {
                var resultTensor = inferResult.FirstElementAs<DenseTensor<long>>();
                return resultTensor.Select(x => (int)x).ToArray();
            }
        }


        /// <summary>
        /// Encodes the tokens.
        /// </summary>
        /// <param name="tokenizedInput">The tokenized input.</param>
        /// <returns></returns>
        public async Task<float[]> EncodeTokensAsync(int[] tokenizedInput)
        {
            // Create input tensor.
            var inputTensor = TensorHelper.CreateTensor(tokenizedInput, new[] { 1, tokenizedInput.Length });
            var inputParameters = CreateInputParameters(NamedOnnxValue.CreateFromTensor("input_ids", inputTensor));

            // Run inference.
            using (var inferResult = await _onnxModelService.RunInferenceAsync(OnnxModelType.TextEncoder, inputParameters))
            {
                var resultTensor = inferResult.FirstElementAs<DenseTensor<float>>();
                return resultTensor.ToArray();
            }
        }


        /// <summary>
        /// Generates the embeds for the input tokens.
        /// </summary>
        /// <param name="inputTokens">The input tokens.</param>
        /// <param name="minimumLength">The minimum length.</param>
        /// <returns></returns>
        private async Task<List<float>> GenerateEmbedsAsync(int[] inputTokens, int minimumLength)
        {
            // If less than minimumLength pad with blank tokens
            if (inputTokens.Length < minimumLength)
                inputTokens = inputTokens.PadWithBlankTokens(minimumLength).ToArray();

            // The CLIP tokenizer only supports 77 tokens, batch process in groups of 77 and concatenate
            var embeddings = new List<float>();
            foreach (var tokenBatch in inputTokens.Batch(Constants.ClipTokenizerTokenLimit))
            {
                var tokens = tokenBatch.PadWithBlankTokens(Constants.ClipTokenizerTokenLimit);
                embeddings.AddRange(await EncodeTokensAsync(tokens.ToArray()));
            }
            return embeddings;
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
