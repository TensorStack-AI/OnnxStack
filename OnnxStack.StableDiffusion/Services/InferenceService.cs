using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
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
        private const int ModelMaxLength = 77;
        private const int EmbeddingsLength = 768;
        private const int BlankTokenValue = 49407;

        private readonly int[] _emptyUncondInput;
        private readonly OnnxStackConfig _configuration;
        private readonly IOnnxModelService _onnxModelService;

        public InferenceService(OnnxStackConfig configuration, IOnnxModelService onnxModelService)
        {
            _configuration = configuration;
            _onnxModelService = onnxModelService;
            _emptyUncondInput = Enumerable.Repeat(BlankTokenValue, ModelMaxLength).ToArray();
        }


        public async Task<DenseTensor<float>> RunInference(StableDiffusionOptions options, SchedulerOptions schedulerConfig)
        {
            // Create random seed if none was set
            options.Seed = options.Seed > 0 ? options.Seed : Random.Shared.Next();

            // Get Scheduler
            var scheduler = GetScheduler(options, schedulerConfig);

            // Get timesteps
            var timesteps = scheduler.SetTimesteps(options.NumInferenceSteps);

            // Preprocess text
            var promptEmbeddings = await GetPromptEmbeddings(options.Prompt, options.NegativePrompt);

            // create latent tensor
            var latents = GenerateLatentSample(options, scheduler.GetInitNoiseSigma());

            // Loop though the timesteps
            foreach (var timestep in timesteps)
            {
                // torch.cat([latents] * 2)
                var latentModelInput = TensorHelper.Duplicate(latents, new[] { 2, 4, options.Height / 8, options.Width / 8 });

                // latent_model_input = scheduler.scale_model_input(latent_model_input, timestep = t)
                latentModelInput = scheduler.ScaleInput(latentModelInput, timestep);

                // Console.WriteLine($"scaled model input {latentModelInput[0]} at step {timestep}. Max {latentModelInput.Max()} Min {latentModelInput.Min()}");
                var input = CreateUnetModelInput(promptEmbeddings, latentModelInput, timestep);

                // Run Inference
                using (var inferResult = await _onnxModelService.RunInferenceAsync(OnnxModelType.Unet, input))
                {
                    var resultTensor = inferResult.FirstElementAs<DenseTensor<float>>();

                    // Split tensors from 2,4,64,64 to 1,4,64,64
                    var splitTensors = TensorHelper.SplitTensor(resultTensor, new[] { 1, 4, options.Height / 8, options.Width / 8 });
                    var noisePred = splitTensors.Item1;
                    var noisePredText = splitTensors.Item2;

                    // Perform guidance
                    noisePred = TensorHelper.PerformGuidance(noisePred, noisePredText, options.GuidanceScale);

                    // LMS Scheduler Step
                    latents = scheduler.Step(noisePred, timestep, latents);
                    //Console.WriteLine($"latents result after step {timestep} min {latents.Min()} max {latents.Max()}");
                }
            }

            // Scale and decode the image latents with vae.
            // latents = 1 / 0.18215 * latents
            latents = TensorHelper.MultipleTensorByFloat(latents, 1.0f / 0.18215f);
          
            // Decode Latents
            return await DecodeLatents(options, latents);
        }


        public async Task<int[]> GetTokens(string text)
        {
            var inputTensor = new DenseTensor<string>(new string[] { text }, new int[] { 1 });
            var inputString = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("string_input", inputTensor)
            };

            // Create an InferenceSession from the onnx clip tokenizer.
            // Run session and send the input data in to get inference output. 
            using (var inferResult = await _onnxModelService.RunInferenceAsync(OnnxModelType.Tokenizer, inputString))
            {
                var resultTensor = inferResult.FirstElementAs<DenseTensor<long>>();
                Console.WriteLine(string.Join(" ", resultTensor));

                // Cast inputIds to Int32
                var inputTokenIds = resultTensor.Select(x => (int)x);
                if (resultTensor.Length < ModelMaxLength)
                {
                    // Pad array with 49407 until length is modelMaxLength
                    inputTokenIds = inputTokenIds.Concat(_emptyUncondInput.Take(ModelMaxLength - (int)resultTensor.Length));
                }
                return inputTokenIds.ToArray();
            }
        }


        private async Task<DenseTensor<float>> GetPromptEmbeddings(string prompt, string negativePrompt)
        {
            // Concat promptEmbeddings and negativePromptEmbeddings
            var promptEmbeddings = await GetTextEmbeddings(prompt);
            var negativePromptEmbeddings = await GetTextEmbeddings(negativePrompt, true);
            var textEmbeddings = new DenseTensor<float>(new[] { 2, ModelMaxLength, EmbeddingsLength });
            for (var i = 0; i < promptEmbeddings.Length; i++)
            {
                textEmbeddings[0, i / EmbeddingsLength, i % EmbeddingsLength] = negativePromptEmbeddings.GetValue(i);
                textEmbeddings[1, i / EmbeddingsLength, i % EmbeddingsLength] = promptEmbeddings.GetValue(i);
            }
            return textEmbeddings;
        }


        private async Task<DenseTensor<float>> GetTextEmbeddings(string text, bool allowEmpty = false)
        {
           var tokens = string.IsNullOrEmpty(text) && allowEmpty
                ? _emptyUncondInput
                : await GetTokens(text);
            return await EncodeTokens(tokens);
        }


        private async Task<DenseTensor<float>> EncodeTokens(int[] tokenizedInput)
        {
            // Create input tensor.
            var input_ids = TensorHelper.CreateTensor(tokenizedInput, new[] { 1, tokenizedInput.Length });
            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_ids", input_ids) };

            // Run inference.
            using (var inferResult = await _onnxModelService.RunInferenceAsync(OnnxModelType.TextEncoder, input))
            {
                var resultTensor = inferResult.FirstElementAs<DenseTensor<float>>();

                // Clone output so it does not get disposed
                return resultTensor.ToDenseTensor();
            }
        }


        private async Task<DenseTensor<float>> DecodeLatents(StableDiffusionOptions options, DenseTensor<float> latents)
        {
            var decoderInput = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("latent_sample", latents)
            };

            using (var inferResult = await _onnxModelService.RunInferenceAsync(OnnxModelType.VaeDecoder, decoderInput))
            {
                var resultTensor = inferResult.FirstElementAs<DenseTensor<float>>();
                if (_configuration.IsSafetyModelEnabled)
                {
                    // Check if image contains NSFW content, if it does return empty tensor (grey image)
                    if (!await IsImageSafe(options, resultTensor))
                        return resultTensor.CloneEmpty().ToDenseTensor();
                }

                // Clone output so it does not get disposed
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
            var inputImagesTensor = TensorHelper.ReorderTensor(inputTensor, new[] { 1, 224, 224, 3 });

            var input = new List<NamedOnnxValue>
            {
                //batch channel height width
                 NamedOnnxValue.CreateFromTensor("clip_input", inputTensor),

                 //batch, height, width, channel
                 NamedOnnxValue.CreateFromTensor("images", inputImagesTensor)
            };

            // Run session and send the input data in to get inference output. 
            using (var inferResult = await _onnxModelService.RunInferenceAsync(OnnxModelType.SafetyModel, input))
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


        private static List<NamedOnnxValue> CreateUnetModelInput(DenseTensor<float> encoderHiddenStates, DenseTensor<float> sample, long timeStep)
        {
            return new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates),
                NamedOnnxValue.CreateFromTensor("sample", sample),
                NamedOnnxValue.CreateFromTensor("timestep", new DenseTensor<long>(new long[] { timeStep }, new int[] { 1 }))
            };
        }


        private static DenseTensor<float> GenerateLatentSample(StableDiffusionOptions options, float initNoiseSigma)
        {
            var random = new Random(options.Seed);
            return TensorHelper.GetRandomTensor(random, new[] { 1, 4, options.Height / 8, options.Width / 8 }, initNoiseSigma);
        }


        private static SchedulerBase GetScheduler(StableDiffusionOptions options, SchedulerOptions schedulerConfig)
        {
            return options.SchedulerType switch
            {
                SchedulerType.LMSScheduler => new LMSScheduler(options, schedulerConfig),
                SchedulerType.EulerAncestralScheduler => new EulerAncestralScheduler(options, schedulerConfig),
                _ => default
            };
        }
    }
}
