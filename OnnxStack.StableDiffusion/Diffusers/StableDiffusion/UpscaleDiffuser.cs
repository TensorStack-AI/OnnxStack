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
using OnnxStack.StableDiffusion.Schedulers.StableDiffusion;
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
            using (var lowResScheduler = new DDPMScheduler())
            using (var scheduler = GetScheduler(promptOptions, schedulerOptions))
            {
                // Should we perform classifier free guidance
                var performGuidance = schedulerOptions.GuidanceScale > 1.0f;

                // Process prompts
                var promptEmbeddings = await _promptService.CreatePromptAsync(modelOptions, promptOptions, performGuidance);

                // Get timesteps
                var timesteps = GetTimesteps(promptOptions, schedulerOptions, scheduler);

                // Create latent sample
                var latents = PrepareLatents(modelOptions, promptOptions, schedulerOptions, scheduler, timesteps);

                // Create Image Tensor
                var image = promptOptions.InputImage.ToDenseTensor(new[] { 1, 3, schedulerOptions.Height, schedulerOptions.Width });

                // Add noise to image
                var noise = lowResScheduler.CreateRandomSample(image.Dimensions);
                image = lowResScheduler.AddNoise(image, noise, new[] { schedulerOptions.NoiseLevel });
                var noiseLevelTensor = new DenseTensor<long>(new[] { (long)schedulerOptions.NoiseLevel }, new[] { 1 });
                if (performGuidance)
                {
                    image = image.Repeat(2);
                    noiseLevelTensor = noiseLevelTensor.Repeat(2);
                }

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
                    inputTensor = ConcatenateLatents(inputTensor, image);


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
                        latents = scheduler.Step(noisePred, timestep, latents).Result;
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


        /// <summary>
        /// Decodes the latents.
        /// </summary>
        /// <param name="model"></param>
        /// <param name="prompt"></param>
        /// <param name="options">The options.</param>
        /// <param name="latents">The latents.</param>
        /// <returns></returns>
        protected override async Task<DenseTensor<float>> DecodeLatents(IModelOptions model, PromptOptions prompt, SchedulerOptions options, DenseTensor<float> latents)
        {
            var timestamp = _logger?.LogBegin("Begin...");

            // Scale and decode the image latents with vae.
            latents = latents.MultiplyBy(1.0f / model.ScaleFactor);

            var images = prompt.BatchCount > 1
                ? latents.Split(prompt.BatchCount)
                : new[] { latents };
            var imageTensors = new List<DenseTensor<float>>();
            foreach (var image in images)
            {
                var inputNames = _onnxModelService.GetInputNames(model, OnnxModelType.Vae);
                var inputParameters = CreateInputParameters(NamedOnnxValue.CreateFromTensor(inputNames[0], image));

                // Run inference.
                using (var inferResult = await _onnxModelService.RunInferenceAsync(model, OnnxModelType.Vae, inputParameters))
                {
                    var resultTensor = inferResult.FirstElementAs<DenseTensor<float>>();
                    imageTensors.Add(resultTensor.ToDenseTensor());
                }
            }

            var result = prompt.BatchCount > 1
                ? imageTensors.Join()
                : imageTensors.FirstOrDefault();
            _logger?.LogEnd("End", timestamp);
            return result;
        }


        /// <summary>
        /// Creates the unet input parameters.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="inputTensor">The input tensor.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <param name="noiseLevel">The noise level.</param>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        private IReadOnlyList<NamedOnnxValue> CreateUnetInputParams(IModelOptions model, DenseTensor<float> inputTensor, DenseTensor<float> promptEmbeddings, DenseTensor<long> noiseLevel, int timestep)
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
            return scheduler.Timesteps;
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
        protected override DenseTensor<float> PrepareLatents(IModelOptions model, PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps)
        {
            return scheduler.CreateRandomSample(new[] { prompt.BatchCount, 4, options.Height, options.Width });
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
    }
}
