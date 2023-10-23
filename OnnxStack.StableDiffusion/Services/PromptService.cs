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
using System.Collections.Immutable;

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
        public async Task<DenseTensor<float>> CreatePromptAsync(IModelOptions model, string prompt, string negativePrompt)
        {
            // Tokenize Prompt and NegativePrompt
            var promptTokens = await DecodeTextAsync(model, prompt);
            var negativePromptTokens = await DecodeTextAsync(model, negativePrompt);
            var maxPromptTokenCount = Math.Max(promptTokens.Length, negativePromptTokens.Length);

            Console.WriteLine($"Prompt -   Length: {prompt.Length}, Tokens: {promptTokens.Length}");
            Console.WriteLine($"N-Prompt - Length: {negativePrompt?.Length}, Tokens: {negativePromptTokens.Length}");

            // Generate embeds for tokens
            var promptEmbeddings = await GenerateEmbedsAsync(model, promptTokens, maxPromptTokenCount);
            var negativePromptEmbeddings = await GenerateEmbedsAsync(model, negativePromptTokens, maxPromptTokenCount);

            // Calculate embeddings
            var textEmbeddings = new DenseTensor<float>(new[] { 2, promptEmbeddings.Count / model.EmbeddingsLength, model.EmbeddingsLength });
            for (var i = 0; i < promptEmbeddings.Count; i++)
            {
                textEmbeddings[0, i / model.EmbeddingsLength, i % model.EmbeddingsLength] = negativePromptEmbeddings[i];
                textEmbeddings[1, i / model.EmbeddingsLength, i % model.EmbeddingsLength] = promptEmbeddings[i];
            }
            return textEmbeddings;
        }


        /// <summary>
        /// Tokenizes the input string
        /// </summary>
        /// <param name="inputText">The input text.</param>
        /// <returns>Tokens generated for the specified text input</returns>
        public async Task<int[]> DecodeTextAsync(IModelOptions model, string inputText)
        {
            if (string.IsNullOrEmpty(inputText))
                return Array.Empty<int>();

            // Create input tensor.
            var inputNames = _onnxModelService.GetInputNames(model, OnnxModelType.Tokenizer);
            var inputTensor = new DenseTensor<string>(new string[] { inputText }, new int[] { 1 });
            var inputParameters = CreateInputParameters(NamedOnnxValue.CreateFromTensor(inputNames[0], inputTensor));

            // Run inference.
            using (var inferResult = await _onnxModelService.RunInferenceAsync(model, OnnxModelType.Tokenizer, inputParameters))
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
        public async Task<float[]> EncodeTokensAsync(IModelOptions model, int[] tokenizedInput)
        {
            // Create input tensor.
            var inputNames = _onnxModelService.GetInputNames(model, OnnxModelType.TextEncoder);
            var inputTensor = TensorHelper.CreateTensor(tokenizedInput, new[] { 1, tokenizedInput.Length });
            var inputParameters = CreateInputParameters(NamedOnnxValue.CreateFromTensor(inputNames[0], inputTensor));

            // Run inference.
            using (var inferResult = await _onnxModelService.RunInferenceAsync(model, OnnxModelType.TextEncoder, inputParameters))
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
        private async Task<List<float>> GenerateEmbedsAsync(IModelOptions model, int[] inputTokens, int minimumLength)
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
            return embeddings;
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
