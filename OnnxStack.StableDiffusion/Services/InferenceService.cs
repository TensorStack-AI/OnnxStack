using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Schedulers;
using OnnxStack.StableDiffusion.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using OnnxStack.Core.Services;

namespace OnnxStack.StableDiffusion.Services
{
    public class InferenceService : IInferenceService
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

        public Tensor<float> RunInference(StableDiffusionOptions options, SchedulerOptions schedulerConfig)
        {
            // Create random seed if none was set
            options.Seed = options.Seed > 0 ? options.Seed : Random.Shared.Next();

            // Get Scheduler
            var scheduler = GetScheduler(options, schedulerConfig);

            // Get timesteps
            var timesteps = scheduler.SetTimesteps(options.NumInferenceSteps);

            // Preprocess text
            var textEmbeddings = PreprocessText(options.Prompt, options.NegativePrompt);

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
                var input = CreateUnetModelInput(textEmbeddings, latentModelInput, timestep);

                // Run Inference
                using (var output = _onnxModelService.RunInference( OnnxModelType.Unet, input))
                {
                    var outputTensor = output.FirstElementAs<DenseTensor<float>>();

                    // Split tensors from 2,4,64,64 to 1,4,64,64
                    var splitTensors = TensorHelper.SplitTensor(outputTensor, new[] { 1, 4, options.Height / 8, options.Width / 8 });
                    var noisePred = splitTensors.Item1;
                    var noisePredText = splitTensors.Item2;

                    // Perform guidance
                    noisePred = PerformGuidance(noisePred, noisePredText, options.GuidanceScale);

                    // LMS Scheduler Step
                    latents = scheduler.Step(noisePred, timestep, latents);
                    //Console.WriteLine($"latents result after step {timestep} min {latents.Min()} max {latents.Max()}");
                }
            }

            // Scale and decode the image latents with vae.
            // latents = 1 / 0.18215 * latents
            latents = TensorHelper.MultipleTensorByFloat(latents, 1.0f / 0.18215f);
            var decoderInput = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("latent_sample", latents)
            };

            // Decode Latents
            using (var decoderOutput = _onnxModelService.RunInference( OnnxModelType.VaeDecoder, decoderInput))
            {
                var imageResultTensor = decoderOutput.FirstElementAs<Tensor<float>>();
                if (_configuration.IsSafetyModelEnabled)
                {
                    // Check if image contains NSFW content, if it does return empty tensor (grey image)
                    if (!IsImageSafe(options, imageResultTensor))
                        return imageResultTensor.CloneEmpty();
                }

                // Clone output so it does not get disposed
                return imageResultTensor.Clone();
            }
        }


        public DenseTensor<float> PreprocessText(string prompt, string negativePrompt)
        {
            // Load the tokenizer and text encoder to tokenize and encode the text.
            var textTokenized = TokenizeText(prompt);
            var textPromptEmbeddings = TextEncoder(textTokenized);

            // Create uncond_input of blank tokens
            var uncondInputTokens = string.IsNullOrEmpty(negativePrompt)
                ? _emptyUncondInput
                : TokenizeText(negativePrompt);
            var uncondEmbedding = TextEncoder(uncondInputTokens);

            // Concat textEmeddings and uncondEmbedding
            var textEmbeddings = new DenseTensor<float>(new[] { 2, ModelMaxLength, EmbeddingsLength });
            for (var i = 0; i < textPromptEmbeddings.Length; i++)
            {
                textEmbeddings[0, i / EmbeddingsLength, i % EmbeddingsLength] = uncondEmbedding.GetValue(i);
                textEmbeddings[1, i / EmbeddingsLength, i % EmbeddingsLength] = textPromptEmbeddings.GetValue(i);
            }
            return textEmbeddings;
        }

        public int[] TokenizeText(string text)
        {
            var inputTensor = new DenseTensor<string>(new string[] { text }, new int[] { 1 });
            var inputString = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("string_input", inputTensor)
            };

            // Create an InferenceSession from the onnx clip tokenizer.
            // Run session and send the input data in to get inference output. 
            using (var tokens = _onnxModelService.RunInference(  OnnxModelType.Tokenizer, inputString))
            {
                var resultTensor = tokens.FirstElementAs<Tensor<long>>();
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


        private List<NamedOnnxValue> CreateUnetModelInput(Tensor<float> encoderHiddenStates, Tensor<float> sample, long timeStep)
        {
            return new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates),
                NamedOnnxValue.CreateFromTensor("sample", sample),
                NamedOnnxValue.CreateFromTensor("timestep", new DenseTensor<long>(new long[] { timeStep }, new int[] { 1 }))
            };
        }


        private Tensor<float> GenerateLatentSample(StableDiffusionOptions options, float initNoiseSigma)
        {
            var random = new Random(options.Seed);
            return TensorHelper.GetRandomTensor(random, new[] { 1, 4, options.Height / 8, options.Width / 8 }, initNoiseSigma);
        }

        private Tensor<float> PerformGuidance(Tensor<float> noisePred, Tensor<float> noisePredText, double guidanceScale)
        {
            for (int i = 0; i < noisePred.Dimensions[0]; i++)
            {
                for (int j = 0; j < noisePred.Dimensions[1]; j++)
                {
                    for (int k = 0; k < noisePred.Dimensions[2]; k++)
                    {
                        for (int l = 0; l < noisePred.Dimensions[3]; l++)
                        {
                            noisePred[i, j, k, l] = noisePred[i, j, k, l] + (float)guidanceScale * (noisePredText[i, j, k, l] - noisePred[i, j, k, l]);
                        }
                    }
                }
            }
            return noisePred;
        }

        private Tensor<float> TextEncoder(int[] tokenizedInput)
        {
            // Create input tensor.
            var input_ids = TensorHelper.CreateTensor(tokenizedInput, new[] { 1, tokenizedInput.Length });
            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_ids", input_ids) };

            // Run inference.
            using (var encoded = _onnxModelService.RunInference( OnnxModelType.TextEncoder, input))
            {
                return encoded.FirstElementAs<DenseTensor<float>>().Clone();
            }
        }



        private SchedulerBase GetScheduler(StableDiffusionOptions options, SchedulerOptions schedulerConfig)
        {
            return options.SchedulerType switch
            {
                SchedulerType.LMSScheduler => new LMSScheduler(options, schedulerConfig),
                SchedulerType.EulerAncestralScheduler => new EulerAncestralScheduler(options, schedulerConfig),
                _ => default
            };
        }


        /// <summary>
        /// Determines whether the specified result image is not NSFW.
        /// </summary>
        /// <param name="resultImage">The result image.</param>
        /// <param name="config">The configuration.</param>
        /// <returns>
        ///   <c>true</c> if the specified result image is safe; otherwise, <c>false</c>.
        /// </returns>
        private bool IsImageSafe(StableDiffusionOptions options, Tensor<float> resultImage)
        {
            //clip input
            var inputTensor = ClipImageFeatureExtractor(options, resultImage);

            //images input
            var inputImagesTensor = ReorderTensor(inputTensor, new[] { 1, 224, 224, 3 });

            var input = new List<NamedOnnxValue>
            {
                //batch channel height width
                 NamedOnnxValue.CreateFromTensor("clip_input", inputTensor),

                 //batch, height, width, channel
                 NamedOnnxValue.CreateFromTensor("images", inputImagesTensor)
            };

            // Run session and send the input data in to get inference output. 
            using (var output = _onnxModelService.RunInference( OnnxModelType.SafetyModel, input))
            {
                var result = output.LastElementAs<IEnumerable<bool>>();
                return !result.First();
            }
        }


        /// <summary>
        /// Reorders the tensor.
        /// </summary>
        /// <param name="inputTensor">The input tensor.</param>
        /// <returns></returns>
        private DenseTensor<float> ReorderTensor(Tensor<float> inputTensor, ReadOnlySpan<int> dimensions)
        {
            //reorder from batch channel height width to batch height width channel
            var inputImagesTensor = new DenseTensor<float>(dimensions);
            for (int y = 0; y < inputTensor.Dimensions[2]; y++)
            {
                for (int x = 0; x < inputTensor.Dimensions[3]; x++)
                {
                    inputImagesTensor[0, y, x, 0] = inputTensor[0, 0, y, x];
                    inputImagesTensor[0, y, x, 1] = inputTensor[0, 1, y, x];
                    inputImagesTensor[0, y, x, 2] = inputTensor[0, 2, y, x];
                }
            }
            return inputImagesTensor;
        }


        /// <summary>
        /// Image feature extractor.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns></returns>
        private DenseTensor<float> ClipImageFeatureExtractor(StableDiffusionOptions options, Tensor<float> imageTensor)
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
    }
}
