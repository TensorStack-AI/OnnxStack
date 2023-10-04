using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Schedulers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;


namespace OnnxStack.StableDiffusion.Services
{
    public sealed class SchedulerService : ISchedulerService
    {
        private readonly OnnxStackConfig _configuration;
        private readonly IOnnxModelService _onnxModelService;
        private readonly IPromptService _promptService;

        /// <summary>
        /// Initializes a new instance of the <see cref="SchedulerService"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="onnxModelService">The onnx model service.</param>
        public SchedulerService(OnnxStackConfig configuration, IOnnxModelService onnxModelService, IPromptService promptService)
        {
            _configuration = configuration;
            _onnxModelService = onnxModelService;
            _promptService = promptService;
        }


        /// <summary>
        /// Runs the Stable Diffusion inference.
        /// </summary>
        /// <param name="promptOptions">The options.</param>
        /// <param name="schedulerOptions">The scheduler configuration.</param>
        /// <returns></returns>
        public async Task<DenseTensor<float>> RunAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions)
        {
            // Create random seed if none was set
            schedulerOptions.Seed = schedulerOptions.Seed > 0 ? schedulerOptions.Seed : Random.Shared.Next();
            Console.WriteLine($"Scheduler: {promptOptions.SchedulerType}, Size: {schedulerOptions.Width}x{schedulerOptions.Height}, Seed: {schedulerOptions.Seed}, Steps: {schedulerOptions.InferenceSteps}, Guidance: {schedulerOptions.GuidanceScale}");

            // Get Scheduler
            using (var scheduler = GetScheduler(promptOptions, schedulerOptions))
            {
                // Process prompts
                var promptEmbeddings = await _promptService.CreatePromptAsync(promptOptions.Prompt, promptOptions.NegativePrompt);

                // Get timesteps
                var timesteps = GetTimesteps(promptOptions, schedulerOptions, scheduler);

                // Create latent sample
                var latentSample = PrepareLatents(promptOptions, schedulerOptions, scheduler, timesteps);

                // Loop though the timesteps
                var step = 0;
                foreach (var timestep in timesteps)
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
                        // ImageHelpers.TensorToImageDebug(latentSample, 64, $@"Examples\StableDebug\Latent_{step}.png");
                    }

                    Console.WriteLine($"Step: {++step}/{timesteps.Count}");
                }

                // Decode Latents
                return await DecodeLatents(schedulerOptions, latentSample);
            }
        }

        private IReadOnlyList<int> GetTimesteps(PromptOptions prompt, SchedulerOptions options, IScheduler scheduler)
        {
            if (!prompt.HasInputImage)
                return scheduler.Timesteps;

            // Image2Image we narrow step the range by the Strength
            var inittimestep = Math.Min((int)(options.InferenceSteps * options.Strength), options.InferenceSteps);
            var start = Math.Max(options.InferenceSteps - inittimestep, 0);
            return scheduler.Timesteps.Skip(start).ToList();
        }

        /// <summary>
        /// Prepares the latents for inference.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <returns></returns>
        private DenseTensor<float> PrepareLatents(PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps)
        {
            // If we dont have an initial image create random sample
            if (!prompt.HasInputImage)
                return scheduler.CreateRandomSample(options.GetScaledDimension(), scheduler.InitNoiseSigma);

            // Image input, decode, add noise, return as latent 0
            var imageTensor = ImageHelpers.TensorFromImage(prompt.InputImage, options.Width, options.Height);
            var inputParameters = CreateInputParameters(NamedOnnxValue.CreateFromTensor("sample", imageTensor));
            using (var inferResult = _onnxModelService.RunInference(OnnxModelType.VaeEncoder, inputParameters))
            {
                var sample = inferResult.FirstElementAs<DenseTensor<float>>();
                var noisySample = sample
                    .AddTensors(scheduler.CreateRandomSample(sample.Dimensions, options.InitialNoiseLevel))
                    .MultipleTensorByFloat(Constants.ModelScaleFactor);
                var noise = scheduler.CreateRandomSample(sample.Dimensions);
                return scheduler.AddNoise(noisySample, noise, timesteps);
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
            latents = latents.MultipleTensorByFloat(1.0f / Constants.ModelScaleFactor);

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
            using (var image = ImageHelpers.TensorToImage(imageTensor, options.Width, options.Height))
            {
                // Resize image
                ImageHelpers.Resize(image, 224, 224);

                // Preprocess image
                var input = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
                var mean = new[] { 0.485f, 0.456f, 0.406f };
                var stddev = new[] { 0.229f, 0.224f, 0.225f };
                image.ProcessPixelRows(img =>
                {
                    for (int y = 0; y < image.Height; y++)
                    {
                        var pixelSpan = img.GetRowSpan(y);
                        for (int x = 0; x < image.Width; x++)
                        {
                            input[0, 0, y, x] = (pixelSpan[x].R / 255f - mean[0]) / stddev[0];
                            input[0, 1, y, x] = (pixelSpan[x].G / 255f - mean[1]) / stddev[1];
                            input[0, 2, y, x] = (pixelSpan[x].B / 255f - mean[2]) / stddev[2];
                        }
                    }
                });
                return input;
            }
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
    }
}
