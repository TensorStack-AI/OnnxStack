using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using SixLabors.ImageSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers.StableDiffusion
{
    public sealed class UpscaleDiffuser : StableDiffusionDiffuser
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="UpscaleDiffuser"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="onnxModelService">The onnx model service.</param>
        public UpscaleDiffuser(IOnnxModelService onnxModelService, IPromptService promptService, ILogger<StableDiffusionDiffuser> logger)
            : base(onnxModelService, promptService, logger)
        {
        }


        /// <summary>
        /// Gets the type of the diffuser.
        /// </summary>
        public override DiffuserType DiffuserType => DiffuserType.ImageUpscale;


        /// <summary>
        /// Run the stable diffusion loop
        /// </summary>
        /// <param name="modelOptions"></param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="progress">The progress.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async Task<DenseTensor<float>> DiffuseAsync(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default)
        {
            // Create random seed if none was set
            schedulerOptions.Seed = schedulerOptions.Seed > 0 ? schedulerOptions.Seed : Random.Shared.Next();

            var diffuseTime = _logger?.LogBegin("Begin...");
            _logger?.Log($"Model: {modelOptions.Name}, Pipeline: {modelOptions.PipelineType}, Diffuser: {promptOptions.DiffuserType}, Scheduler: {promptOptions.SchedulerType}");

            // Get Scheduler
            using (var scheduler = GetScheduler(promptOptions, schedulerOptions))
            {
                // Should we perform classifier free guidance
                var performGuidance = schedulerOptions.GuidanceScale > 1.0f;

                // Process prompts
                var promptEmbeddings = await _promptService.CreatePromptAsync(modelOptions, promptOptions, performGuidance);

                // Get timesteps
                var timesteps = GetTimesteps(promptOptions, schedulerOptions, scheduler);

                // Create latent sample
                var latentsOriginal = PrepareLatents(modelOptions, promptOptions, schedulerOptions, scheduler, timesteps);

                long noiseLevel = 20;

                // Generate some noise
                var noise = scheduler.CreateRandomSample(latentsOriginal.Dimensions, noiseLevel);

                // Add noise to original latent
                var latents = scheduler.AddNoise(latentsOriginal, noise, timesteps);

                // noiseLevel = noiseLevel * latents.Dimensions[0];
                var noiseLevelTensor= new DenseTensor<long>(Enumerable.Range(0, latents.Dimensions[0]).Select(x => noiseLevel).ToArray(), new[] { 1 });

                // Loop though the timesteps
                var step = 0;
                foreach (var timestep in timesteps)
                {
                    step++;
                    var stepTime = Stopwatch.GetTimestamp();
                    cancellationToken.ThrowIfCancellationRequested();

                    // Create input tensor.
                    var inputLatent = performGuidance
                        ? latents.Repeat(2)
                        : latents;
                    var inputTensor = scheduler.ScaleInput(inputLatent, timestep);

                    // Create Input Parameters
                    var inputParameters = CreateUnetInputParams(modelOptions, inputTensor, promptEmbeddings, noiseLevelTensor, timestep);

                    // Run Inference
                    using (var inferResult = await _onnxModelService.RunInferenceAsync(modelOptions, OnnxModelType.Unet, inputParameters))
                    {
                        var noisePred = inferResult.FirstElementAs<DenseTensor<float>>();

                        // Perform guidance
                        if (performGuidance)
                            noisePred = PerformGuidance(noisePred, schedulerOptions.GuidanceScale);

                        // Scheduler Step
                        var steplatents = scheduler.Step(noisePred, timestep, latents).Result;

                        // Add noise to original latent
                        latents = scheduler.AddNoise(latentsOriginal, noise, new[] { timestep });
                    }

                    progressCallback?.Invoke(step, timesteps.Count);
                    _logger?.LogEnd(LogLevel.Debug, $"Step {step}/{timesteps.Count}", stepTime);
                }

                // Decode Latents
                var result = await DecodeLatents(modelOptions, promptOptions, schedulerOptions, latents);
                _logger?.LogEnd($"End", diffuseTime);
                return result;
            }
        }

        protected IReadOnlyList<NamedOnnxValue> CreateUnetInputParams(IModelOptions model, DenseTensor<float> inputTensor, DenseTensor<float> promptEmbeddings, DenseTensor<long> noiseLevel, int timestep)
        {
            var inputNames = _onnxModelService.GetInputNames(model, OnnxModelType.Unet);
            var inputMetaData = _onnxModelService.GetInputMetadata(model, OnnxModelType.Unet);

            // Some models support Long or Float, could be more but fornow just support these 2
            var timesepMetaKey = inputNames[1];
            var timestepMetaData = inputMetaData[timesepMetaKey];
            var timestepNamedOnnxValue = timestepMetaData.ElementDataType == TensorElementType.Int64
                ? NamedOnnxValue.CreateFromTensor(timesepMetaKey, new DenseTensor<long>(new long[] { timestep }, new int[] { 1 }))
                : NamedOnnxValue.CreateFromTensor(timesepMetaKey, new DenseTensor<float>(new float[] { timestep }, new int[] { 1 }));

            return CreateInputParameters(
                 NamedOnnxValue.CreateFromTensor(inputNames[0], inputTensor),
                 timestepNamedOnnxValue,
                 NamedOnnxValue.CreateFromTensor(inputNames[2], promptEmbeddings),
                 NamedOnnxValue.CreateFromTensor(inputNames[3], noiseLevel));
        }


        /// <summary>
        /// Gets the timesteps.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <returns></returns>
        protected override IReadOnlyList<int> GetTimesteps(PromptOptions prompt, SchedulerOptions options, IScheduler scheduler)
        {
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
        protected override DenseTensor<float> PrepareLatents(IModelOptions model, PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps)
        {
            // Image input, decode, add noise, return as latent 0
            var imageTensor = prompt.InputImage.ToDenseTensor(new[] { 1, 3, options.Width, options.Height });
            var inputNames = _onnxModelService.GetInputNames(model, OnnxModelType.VaeEncoder);
            var inputParameters = CreateInputParameters(NamedOnnxValue.CreateFromTensor(inputNames[0], imageTensor));
            using (var inferResult = _onnxModelService.RunInference(model, OnnxModelType.VaeEncoder, inputParameters))
            {
                var sample = inferResult.FirstElementAs<DenseTensor<float>>();
                var scaledSample = sample
                     .Add(scheduler.CreateRandomSample(sample.Dimensions, options.InitialNoiseLevel))
                     .MultiplyBy(model.ScaleFactor)
                     .ToDenseTensor();

                if (prompt.BatchCount > 1)
                    return scaledSample.Repeat(prompt.BatchCount);

                return scaledSample;
            }
        }
    }
}
