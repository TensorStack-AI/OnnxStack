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

namespace OnnxStack.StableDiffusion.Services
{
    public class InferenceService : IInferenceService
    {
        private const int ModelMaxLength = 77;
        private const int EmbeddingsLength = 768;
        private const int BlankTokenValue = 49407;

        private readonly int[] _emptyUncondInput;
        private readonly SessionOptions _sessionOptions;
        private readonly OnnxStackConfig _configuration;
        private readonly InferenceSession _onnxUnetInferenceSession;
        private readonly InferenceSession _onnxTokenizerInferenceSession;
        private readonly InferenceSession _onnxVaeDecoderInferenceSession;
        private readonly InferenceSession _onnxTextEncoderInferenceSession;

        public InferenceService(OnnxStackConfig configuration)
        {
            _configuration = configuration;
            _sessionOptions = _configuration.GetSessionOptions();
            _sessionOptions.RegisterOrtExtensions();
            _emptyUncondInput = Enumerable.Repeat(BlankTokenValue, ModelMaxLength).ToArray();
            _onnxUnetInferenceSession = new InferenceSession(_configuration.OnnxUnetPath, _sessionOptions);
            _onnxTokenizerInferenceSession = new InferenceSession(_configuration.OnnxTokenizerPath, _sessionOptions);
            _onnxVaeDecoderInferenceSession = new InferenceSession(_configuration.OnnxVaeDecoderPath, _sessionOptions);
            _onnxTextEncoderInferenceSession = new InferenceSession(_configuration.OnnxTextEncoderPath, _sessionOptions);
        }



        public Tensor<float> RunInference(StableDiffusionOptions options, SchedulerOptions schedulerConfig)
        {
            // Get Scheduler
            var scheduler = GetScheduler(options, schedulerConfig);

            // Get timesteps
            var timesteps = scheduler.SetTimesteps(options.NumInferenceSteps);

            // Preprocess text
            var textEmbeddings = PreprocessText(options.Prompt, options.NegativePrompt);

            // create latent tensor
            var latents = GenerateLatentSample(options, scheduler.GetInitNoiseSigma());


            foreach (var timestep in timesteps)
            {
                // torch.cat([latents] * 2)
                var latentModelInput = TensorHelper.Duplicate(latents, new[] { 2, 4, options.Height / 8, options.Width / 8 });

                // latent_model_input = scheduler.scale_model_input(latent_model_input, timestep = t)
                latentModelInput = scheduler.ScaleInput(latentModelInput, timestep);

               // Console.WriteLine($"scaled model input {latentModelInput[0]} at step {timestep}. Max {latentModelInput.Max()} Min {latentModelInput.Min()}");
                var input = CreateUnetModelInput(textEmbeddings, latentModelInput, timestep);

                // Run Inference
                using (var output = _onnxUnetInferenceSession.Run(input))
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

            using (var decoderOutput = _onnxVaeDecoderInferenceSession.Run(decoderInput))
            {
                return decoderOutput.FirstElementAs<Tensor<float>>().Clone();
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
            using (var tokens = _onnxTokenizerInferenceSession.Run(inputString))
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
            var latents = new DenseTensor<float>(new[] { 1, 4, options.Height / 8, options.Width / 8 });
            for (int i = 0; i < latents.Length; i++)
            {
                // Generate a random number from a normal distribution with mean 0 and variance 1
                var u1 = random.NextDouble(); // Uniform(0,1) random number
                var u2 = random.NextDouble(); // Uniform(0,1) random number
                var radius = Math.Sqrt(-2.0 * Math.Log(u1)); // Radius of polar coordinates
                var theta = 2.0 * Math.PI * u2; // Angle of polar coordinates
                var standardNormalRand = radius * Math.Cos(theta); // Standard normal random number

                // add noise to latents with * scheduler.init_noise_sigma
                // generate randoms that are negative and positive
                latents.SetValue(i, (float)standardNormalRand * initNoiseSigma);
            }
            return latents;
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
            using (var encoded = _onnxTextEncoderInferenceSession.Run(input))
            {
                return encoded.FirstElementAs<DenseTensor<float>>().Clone();
            }
        }



        private SchedulerBase GetScheduler(StableDiffusionOptions options, SchedulerOptions schedulerConfig)
        {
            return options.SchedulerType switch
            {
                SchedulerType.LMSScheduler => new LMSScheduler(schedulerConfig),
                SchedulerType.EulerAncestralScheduler => new EulerAncestralScheduler(schedulerConfig),
                _ => default
            };
        }
        public void Dispose()
        {
            _sessionOptions.Dispose();
            _onnxUnetInferenceSession.Dispose();
            _onnxTokenizerInferenceSession.Dispose();
            _onnxVaeDecoderInferenceSession.Dispose();
            _onnxTextEncoderInferenceSession.Dispose();
        }
    }
}
