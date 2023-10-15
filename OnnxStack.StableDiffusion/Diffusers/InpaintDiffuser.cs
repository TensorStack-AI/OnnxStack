using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Config;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Helpers;
using SixLabors.ImageSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;


namespace OnnxStack.StableDiffusion.Services
{
    public sealed class InpaintDiffuser : DiffuserBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="InpaintDiffuser"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="onnxModelService">The onnx model service.</param>
        public InpaintDiffuser(IOnnxModelService onnxModelService, IPromptService promptService)
            :base(onnxModelService, promptService)
        {
        }


        /// <summary>
        /// Runs the Stable Diffusion inference.
        /// </summary>
        /// <param name="promptOptions">The options.</param>
        /// <param name="schedulerOptions">The scheduler configuration.</param>
        /// <returns></returns>
        public override async Task<DenseTensor<float>> DiffuseAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions, Action<int, int> progress = null, CancellationToken cancellationToken = default)
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
                    cancellationToken.ThrowIfCancellationRequested();

                    // Create input tensor.
                    var inputTensor = scheduler.ScaleInput(latentSample.Duplicate(schedulerOptions.GetScaledDimension(2)), timestep);

                    var inputNames = _onnxModelService.GetInputNames(OnnxModelType.Unet);
                    var inputParameters = CreateInputParameters(
                         NamedOnnxValue.CreateFromTensor(inputNames[0], inputTensor),
                         NamedOnnxValue.CreateFromTensor(inputNames[1], new DenseTensor<long>(new long[] { timestep }, new int[] { 1 })),
                         NamedOnnxValue.CreateFromTensor(inputNames[2], promptEmbeddings));

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
                    progress?.Invoke(step, timesteps.Count);
                }

                // Decode Latents
                return await DecodeLatents(schedulerOptions, latentSample);
            }
        }

        protected override IReadOnlyList<int> GetTimesteps(PromptOptions prompt, SchedulerOptions options, IScheduler scheduler)
        {
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
        protected override DenseTensor<float> PrepareLatents(PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps)
        {
            // Image input, decode, add noise, return as latent 0
            var imageTensor = prompt.InputImage.ToDenseTensor(options.Width, options.Height);
            var inputNames = _onnxModelService.GetInputNames(OnnxModelType.VaeEncoder);
            var inputParameters = CreateInputParameters(NamedOnnxValue.CreateFromTensor(inputNames[0], imageTensor));
            using (var inferResult = _onnxModelService.RunInference(OnnxModelType.VaeEncoder, inputParameters))
            {
                var sample = inferResult.FirstElementAs<DenseTensor<float>>();
                var noisySample = sample
                    .AddTensors(scheduler.CreateRandomSample(sample.Dimensions, options.InitialNoiseLevel))
                    .MultipleTensorByFloat(_configuration.ScaleFactor);
                var noise = scheduler.CreateRandomSample(sample.Dimensions);
                return scheduler.AddNoise(noisySample, noise, timesteps);
            }
        }

    }
}
