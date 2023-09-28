using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Schedulers;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Services
{
    public sealed class InferenceService : IInferenceService
    {
        private readonly OnnxStackConfig _configuration;
        private readonly IOnnxModelService _onnxModelService;

        /// <summary>
        /// Initializes a new instance of the <see cref="InferenceService"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="onnxModelService">The onnx model service.</param>
        public InferenceService(OnnxStackConfig configuration, IOnnxModelService onnxModelService)
        {
            _configuration = configuration;
            _onnxModelService = onnxModelService;
        }


        /// <summary>
        /// Runs the Stable Diffusion inference.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="schedulerConfig">The scheduler configuration.</param>
        /// <returns></returns>
        public async Task<DenseTensor<float>> RunInferenceAsync(StableDiffusionOptions options, SchedulerOptions schedulerConfig)
        {
            // Create random seed if none was set
            options.Seed = options.Seed > 0 ? options.Seed : Random.Shared.Next();

            var random = new Random(options.Seed); //TODO: Add to StableDiffusionOptions so can be shared with SchedulerBase
            Console.WriteLine($"Scheduler: {options.SchedulerType}, Size: {options.Width}x{options.Height}, Seed: {options.Seed}, Steps: {options.NumInferenceSteps}, Guidance: {options.GuidanceScale}");

            // Get Scheduler
            using (var scheduler = GetScheduler(options, schedulerConfig))
            {

                // Process prompts
                var promptEmbeddings = await CreatePromptEmbeddings(options.Prompt, options.NegativePrompt);

                // Create latent sample
                var latentSample = scheduler.CreateRandomSample(options.GetScaledDimension(), scheduler.InitNoiseSigma);

                // Loop though the timesteps
                var step = 0;
                foreach (var timestep in scheduler.Timesteps)
                {
                    // Create input tensor.
                    var inputTensor = scheduler.ScaleInput(latentSample.Duplicate(options.GetScaledDimension(2)), timestep);
                    var inputParameters = CreateInputParameters(
                         NamedOnnxValue.CreateFromTensor("encoder_hidden_states", promptEmbeddings),
                         NamedOnnxValue.CreateFromTensor("sample", inputTensor),
                         NamedOnnxValue.CreateFromTensor("timestep", new DenseTensor<long>(new long[] { timestep }, new int[] { 1 })));

                    // Run Inference
                    using (var inferResult = await _onnxModelService.RunInferenceAsync(OnnxModelType.Unet, inputParameters))
                    {
                        var resultTensor = inferResult.FirstElementAs<DenseTensor<float>>();

                        // Split tensors from 2,4,(H/8),(W/8) to 1,4,(H/8),(W/8)
                        var splitTensors = resultTensor.SplitTensor(options.GetScaledDimension(), options.GetScaledHeight(), options.GetScaledWidth());
                        var noisePred = splitTensors.Item1;
                        var noisePredText = splitTensors.Item2;

                        // Perform guidance
                        noisePred = noisePred.PerformGuidance(noisePredText, options.GuidanceScale);

                        // LMS Scheduler Step
                        latentSample = scheduler.Step(noisePred, timestep, latentSample);
                    }

                    Console.WriteLine($"Step: {++step}/{scheduler.Timesteps.Count}");
                }

                // Scale and decode the image latents with vae.
                // latents = 1 / 0.18215 * latents
                latentSample = latentSample.MultipleTensorByFloat(1.0f / 0.18215f);

                // Decode Latents
                return await DecodeLatents(options, latentSample);
            }
        }


        /// <summary>
        /// Creates the prompt embeddings.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="negativePrompt">The negative prompt.</param>
        /// <returns>Tensor containing all text embeds generated from the prompt and negative prompt</returns>
        private async Task<DenseTensor<float>> CreatePromptEmbeddings(string prompt, string negativePrompt)
        {
            // Tokenize Prompt and NegativePrompt
            var promptTokens = await TokenizeAsync(prompt);
            var negativePromptTokens = await TokenizeAsync(negativePrompt);
            var maxPromptTokenCount = Math.Max(promptTokens.Length, negativePromptTokens.Length);

            Console.WriteLine($"Prompt -   Length: {prompt.Length}, Tokens: {promptTokens.Length}");
            Console.WriteLine($"N-Prompt - Length: {negativePrompt?.Length}, Tokens: {negativePromptTokens.Length}");

            // Generate embeds for tokens
            var promptEmbeddings = await GenerateEmbeds(promptTokens, maxPromptTokenCount);
            var negativePromptEmbeddings = await GenerateEmbeds(negativePromptTokens, maxPromptTokenCount);

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
        /// Generates the embeds for the input tokens.
        /// </summary>
        /// <param name="inputTokens">The input tokens.</param>
        /// <param name="minimumLength">The minimum length.</param>
        /// <returns></returns>
        private async Task<List<float>> GenerateEmbeds(int[] inputTokens, int minimumLength)
        {
            // If less than minimumLength pad with balnk tokens
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
        /// Tokenizes the input string
        /// </summary>
        /// <param name="inputText">The input text.</param>
        /// <returns>Tokens generated for the specified text input</returns>
        public async Task<int[]> TokenizeAsync(string inputText)
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
        private async Task<float[]> EncodeTokensAsync(int[] tokenizedInput)
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
        /// Decodes the latents.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="latents">The latents.</param>
        /// <returns></returns>
        private async Task<DenseTensor<float>> DecodeLatents(StableDiffusionOptions options, DenseTensor<float> latents)
        {
            var inputParameters = CreateInputParameters(NamedOnnxValue.CreateFromTensor("latent_sample", latents));

            // Run inference.
            using (var inferResult = await _onnxModelService.RunInferenceAsync(OnnxModelType.VaeDecoder, inputParameters))
            {
                var resultTensor = inferResult.FirstElementAs<DenseTensor<float>>();
                if (_configuration.IsSafetyModelEnabled)
                {
                    // Check if image contains NSFW content, 
                    if (!await IsImageSafe(options, resultTensor))
                        return resultTensor.CloneEmpty().ToDenseTensor(); //TODO: blank image?, exception?, null?
                }
                return resultTensor.ToDenseTensor();
            }
        }


        /// <summary>
        /// Determines whether the specified result image is not NSFW.
        /// </summary>
        /// <param name="resultImage">The result image.</param>
        /// <param name="config">The configuration.</param>
        /// <returns>
        ///   <c>true</c> if the specified result image is safe; otherwise, <c>false</c>.
        /// </returns>
        private async Task<bool> IsImageSafe(StableDiffusionOptions options, DenseTensor<float> resultImage)
        {
            //clip input
            var inputTensor = ClipImageFeatureExtractor(options, resultImage);

            //images input
            var inputImagesTensor = inputTensor.ReorderTensor(new[] { 1, 224, 224, 3 });
            var inputParameters = CreateInputParameters(
                NamedOnnxValue.CreateFromTensor("clip_input", inputTensor),
                NamedOnnxValue.CreateFromTensor("images", inputImagesTensor));

            // Run session and send the input data in to get inference output. 
            using (var inferResult = await _onnxModelService.RunInferenceAsync(OnnxModelType.SafetyModel, inputParameters))
            {
                var result = inferResult.LastElementAs<IEnumerable<bool>>();
                return !result.First();
            }
        }


        /// <summary>
        /// Image feature extractor.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns></returns>
        private static DenseTensor<float> ClipImageFeatureExtractor(StableDiffusionOptions options, DenseTensor<float> imageTensor)
        {
            //convert tensor result to image
            var image = new Image<Rgba32>(options.Width, options.Height);

            for (var y = 0; y < options.Height; y++)
            {
                for (var x = 0; x < options.Width; x++)
                {
                    image[x, y] = new Rgba32(
                        (byte)Math.Round(Math.Clamp(imageTensor[0, 0, y, x] / 2 + 0.5, 0, 1) * 255),
                        (byte)Math.Round(Math.Clamp(imageTensor[0, 1, y, x] / 2 + 0.5, 0, 1) * 255),
                        (byte)Math.Round(Math.Clamp(imageTensor[0, 2, y, x] / 2 + 0.5, 0, 1) * 255)
                    );
                }
            }

            // Resize image
            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(224, 224),
                    Mode = ResizeMode.Crop
                });
            });

            // Preprocess image
            var input = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var stddev = new[] { 0.229f, 0.224f, 0.225f };
            for (int y = 0; y < image.Height; y++)
            {
                Span<Rgba32> pixelSpan = image.GetPixelRowSpan(y);

                for (int x = 0; x < image.Width; x++)
                {
                    input[0, 0, y, x] = (pixelSpan[x].R / 255f - mean[0]) / stddev[0];
                    input[0, 1, y, x] = (pixelSpan[x].G / 255f - mean[1]) / stddev[1];
                    input[0, 2, y, x] = (pixelSpan[x].B / 255f - mean[2]) / stddev[2];
                }
            }

            return input;
        }


        /// <summary>
        /// Gets the scheduler.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="schedulerConfig">The scheduler configuration.</param>
        /// <returns></returns>
        private static IScheduler GetScheduler(StableDiffusionOptions options, SchedulerOptions schedulerConfig)
        {
            return options.SchedulerType switch
            {
                SchedulerType.LMSScheduler => new LMSScheduler(options, schedulerConfig),
                SchedulerType.EulerAncestralScheduler => new EulerAncestralScheduler(options, schedulerConfig),
                SchedulerType.DDPMScheduler => new DDPMScheduler(options, schedulerConfig),
                _ => default
            };
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
