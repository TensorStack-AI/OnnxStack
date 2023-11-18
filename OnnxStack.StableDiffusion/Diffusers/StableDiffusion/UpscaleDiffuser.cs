using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Schedulers.StableDiffusion;
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
            _logger?.Log($"Model: {modelOptions.Name}, Pipeline: {modelOptions.PipelineType}, Diffuser: {promptOptions.DiffuserType}, Scheduler: {schedulerOptions.SchedulerType}");

            // Get Scheduler
            using (var lowResScheduler = new DDPMScheduler())
            using (var scheduler = GetScheduler(schedulerOptions))
            {
                // Should we perform classifier free guidance
                var performGuidance = schedulerOptions.GuidanceScale > 1.0f;

                // Process prompts
                var promptEmbeddings = await _promptService.CreatePromptAsync(modelOptions, promptOptions, performGuidance);

                // Get timesteps
                var timesteps = GetTimesteps(schedulerOptions, scheduler);

                // Create latent sample
                var latents = await PrepareLatentsAsync(modelOptions, promptOptions, schedulerOptions, scheduler, timesteps);
                ImageHelpers.TensorToImageDebug(latents, $@"Examples\UpscaleDebug\Latent.png");

                // Create Image Tensor
                var image = promptOptions.InputImage.ToDenseTensor(new[] { 1, 3, schedulerOptions.Height, schedulerOptions.Width });
                ImageHelpers.TensorToImageDebug(image, $@"Examples\UpscaleDebug\Image.png");

                // Add noise to image
                var noise = scheduler.CreateRandomSample(image.Dimensions);
                ImageHelpers.TensorToImageDebug(latents, $@"Examples\UpscaleDebug\Noise.png");

                image = lowResScheduler.AddNoise(image, noise, new[] { schedulerOptions.NoiseLevel });
                ImageHelpers.TensorToImageDebug(image, $@"Examples\UpscaleDebug\NoiseImage.png");

                var noiseLevelTensor = new DenseTensor<long>(new[] { (long)schedulerOptions.NoiseLevel }, new[] { 1 });
                if (performGuidance)
                {
                    image = image.Repeat(2);
                    noiseLevelTensor = noiseLevelTensor.Repeat(2);
                }

                // Get Model metadata
                var metadata = _onnxModelService.GetModelMetadata(modelOptions, OnnxModelType.Unet);

                // Loop though the timesteps
                var step = 0;
                foreach (var timestep in timesteps)
                {
                    step++;
                    var stepTime = Stopwatch.GetTimestamp();
                    cancellationToken.ThrowIfCancellationRequested();

                    // Create input tensor.
                    var inputLatent = performGuidance ? latents.Repeat(2) : latents;
                    var inputTensor = scheduler.ScaleInput(inputLatent, timestep);
                    inputTensor = ConcatenateLatents(inputTensor, image);
                    var timestepTensor = CreateTimestepTensor(timestep);

                    // Create Input Parameters
                    var outputChannels = performGuidance ? 2 : 1;
                    var outputDimension = schedulerOptions.GetScaledDimension(outputChannels);
                    using (var inferenceParameters = new OnnxInferenceParameters(metadata))
                    {
                        inferenceParameters.AddInputTensor(inputTensor);
                        inferenceParameters.AddInputTensor(timestepTensor);
                        inferenceParameters.AddInputTensor(promptEmbeddings);
                        //inferenceParameters.AddInputTensor(noiseLevelTensor);
                        inferenceParameters.AddOutputBuffer(outputDimension);


                        var results = await _onnxModelService.RunInferenceAsync(modelOptions, OnnxModelType.Unet, inferenceParameters);
                        using (var result = results.First())
                        {
                            var noisePred = result.ToDenseTensor();

                            // Perform guidance
                            if (performGuidance)
                                noisePred = PerformGuidance(noisePred, schedulerOptions.GuidanceScale);

                            // Scheduler Step
                            latents = scheduler.Step(noisePred, timestep, latents).Result;

                            ImageHelpers.TensorToImageDebug(latents, $@"Examples\UpscaleDebug\Latent_{step}.png");
                        }
                    }

                    progressCallback?.Invoke(step, timesteps.Count);
                    _logger?.LogEnd(LogLevel.Debug, $"Step {step}/{timesteps.Count}", stepTime);
                }

                return await DecodeLatentsAsync(modelOptions, promptOptions, schedulerOptions, latents);
            }
        }


        /// <summary>
        /// Prepares the latents.
        /// </summary>
        /// <param name="model"></param>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <returns></returns>
        protected override Task<DenseTensor<float>> PrepareLatentsAsync(IModelOptions model, PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps)
        {
            return Task.FromResult(scheduler.CreateRandomSample(new[] { 1, 4, options.Height, options.Width }, scheduler.InitNoiseSigma));
        }


        /// <summary>
        /// Concatenates the latents.
        /// </summary>
        /// <param name="input1">The input1.</param>
        /// <param name="input2">The input2.</param>
        /// <returns></returns>
        private DenseTensor<float> ConcatenateLatents(DenseTensor<float> input1, DenseTensor<float> input2)
        {
            int batch = input1.Dimensions[0];
            int height = input1.Dimensions[2];
            int width = input1.Dimensions[3];
            int channels = input1.Dimensions[1] + input2.Dimensions[1];

            var concatenated = new DenseTensor<float>(new[] { batch, channels, height, width });

            for (int i = 0; i < batch; i++)
            {
                for (int j = 0; j < channels; j++)
                {
                    if (j < input1.Dimensions[1])
                    {
                        // Copy from input1
                        for (int k = 0; k < height; k++)
                        {
                            for (int l = 0; l < width; l++)
                            {
                                concatenated[i, j, k, l] = input1[i, j, k, l];
                            }
                        }
                    }
                    else
                    {
                        // Copy from input2
                        for (int k = 0; k < height; k++)
                        {
                            for (int l = 0; l < width; l++)
                            {
                                concatenated[i, j, k, l] = input2[i, j - input1.Dimensions[1], k, l];
                            }
                        }
                    }
                }
            }
            return concatenated;
        }

        protected override IReadOnlyList<int> GetTimesteps(SchedulerOptions options, IScheduler scheduler)
        {
            return scheduler.Timesteps;
        }

    }
}
