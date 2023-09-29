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
        /// <param name="promptOptions">The options.</param>
        /// <param name="schedulerOptions">The scheduler configuration.</param>
        /// <returns></returns>
        public async Task<DenseTensor<float>> RunInferenceAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions)
        {
            // Create random seed if none was set
            schedulerOptions.Seed = schedulerOptions.Seed > 0 ? schedulerOptions.Seed : Random.Shared.Next();


            Console.WriteLine($"Scheduler: {promptOptions.SchedulerType}, Size: {schedulerOptions.Width}x{schedulerOptions.Height}, Seed: {schedulerOptions.Seed}, Steps: {schedulerOptions.InferenceSteps}, Guidance: {schedulerOptions.GuidanceScale}");

            // Get Scheduler
            using (var scheduler = GetScheduler(promptOptions, schedulerOptions))
            {
                // Process prompts
                var promptEmbeddings = await CreatePromptEmbeddings(promptOptions.Prompt, promptOptions.NegativePrompt);

                // Create latent sample
                var latentSample = PrepareLatents(promptOptions, schedulerOptions, scheduler);

                // Loop though the timesteps
                var step = 0;
                foreach (var timestep in scheduler.Timesteps)
                {
                    // Create input tensor.
                    var inputTensor = scheduler.ScaleInput(latentSample.Duplicate(schedulerOptions.GetScaledDimension(2)), timestep);

                    var inputParameters = CreateInputParameters(
                         NamedOnnxValue.CreateFromTensor("encoder_hidden_states", promptEmbeddings),
                         NamedOnnxValue.CreateFromTensor("sample", inputTensor),
                         NamedOnnxValue.CreateFromTensor("timestep", new DenseTensor<long>(new long[] { timestep }, new int[] { 1 })));

                    // Run Inference
                    using (var inferResult = await _onnxModelService.RunInferenceAsync(OnnxModelType.Unet, inputParameters))
                    {
                        var resultTensor = inferResult.FirstElementAs<DenseTensor<float>>();

                        // Split tensors from 2,4,(H/8),(W/8) to 1,4,(H/8),(W/8)
                        var splitTensors = resultTensor.SplitTensor(schedulerOptions.GetScaledDimension(), schedulerOptions.GetScaledHeight(), schedulerOptions.GetScaledWidth());
                        var noisePred = splitTensors.Item1;
                        var noisePredText = splitTensors.Item2;

                        // Perform guidance
                        noisePred = noisePred.PerformGuidance(noisePredText, schedulerOptions.GuidanceScale);

                        // LMS Scheduler Step
                        latentSample = scheduler.Step(noisePred, timestep, latentSample);
                        //ImageHelpers.TensorToImageDebug(latentSample, 512, $@"Examples\StableDebug\Latent_{step}.png");
                    }

                    Console.WriteLine($"Step: {++step}/{scheduler.Timesteps.Count}");
                }

                // Decode Latents
                return await DecodeLatents(schedulerOptions, latentSample);
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
        private async Task<DenseTensor<float>> DecodeLatents(SchedulerOptions options, DenseTensor<float> latents)
        {
            // Scale and decode the image latents with vae.
            // latents = 1 / 0.18215 * latents
            latents = latents.MultipleTensorByFloat(1.0f / 0.18215f);

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
        private async Task<bool> IsImageSafe(SchedulerOptions options, DenseTensor<float> resultImage)
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
        private static DenseTensor<float> ClipImageFeatureExtractor(SchedulerOptions options, DenseTensor<float> imageTensor)
        {
            //convert tensor result to image
            var image = new Image<Rgba32>(options.Width, options.Height);

            for (var y = 0; y < options.Height; y++)
            {
                for (var x = 0; x < options.Width; x++)
                {
                    image[x, y] = new Rgba32(
                        ImageHelpers.CalculateByte(imageTensor, 0, y, x),
                        ImageHelpers.CalculateByte(imageTensor, 1, y, x),
                        ImageHelpers.CalculateByte(imageTensor, 2, y, x)
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
        private static IScheduler GetScheduler(PromptOptions prompt, SchedulerOptions options)
        {
            return prompt.SchedulerType switch
            {
                SchedulerType.LMSScheduler => new LMSScheduler(options),
                SchedulerType.EulerAncestralScheduler => new EulerAncestralScheduler(options),
                SchedulerType.DDPMScheduler => new DDPMScheduler(options),
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


        /// <summary>
        /// Prepares the latents for inference.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <returns></returns>
        public DenseTensor<float> PrepareLatents(PromptOptions prompt, SchedulerOptions options, IScheduler scheduler)
        {
            // If we dont have an initial image create random sample
            if (!prompt.HasInputImage)
                return scheduler.CreateRandomSample(options.GetScaledDimension(), scheduler.InitNoiseSigma);

            // We have an initial image, resize and encode to latent
            using (Image<Rgb24> image = Image.Load<Rgb24>(prompt.InputImage))
            {
                image.Mutate(x =>
                {
                    x.Resize(new ResizeOptions
                    {
                        Size = new Size(options.Width, options.Height),
                        Mode = ResizeMode.Crop
                    });
                });

                var mean = new[] { 0.485f, 0.456f, 0.406f };
                var imageArray = new DenseTensor<float>(new[] { 1, 3, options.Width, options.Height });
                for (int x = 0; x < options.Width; x++)
                {
                    for (int y = 0; y < options.Height; y++)
                    {
                        var pixelSpan = image.GetPixelRowSpan(y);
                        imageArray[0, 0, y, x] = (pixelSpan[x].R / 255.0f) - mean[0];
                        imageArray[0, 1, y, x] = (pixelSpan[x].G / 255.0f) - mean[1];
                        imageArray[0, 2, y, x] = (pixelSpan[x].B / 255.0f) - mean[2];
                    }
                }

                var inputParameters = CreateInputParameters(NamedOnnxValue.CreateFromTensor("sample", imageArray));
                using (var inferResult = _onnxModelService.RunInference(OnnxModelType.VaeEncoder, inputParameters))
                {
                    var result = inferResult.FirstElementAs<DenseTensor<float>>();
                    return scheduler.AddNoise(result, scheduler.CreateRandomSample(options.GetScaledDimension(), scheduler.InitNoiseSigma));
                }
            }
        }
    }
}
