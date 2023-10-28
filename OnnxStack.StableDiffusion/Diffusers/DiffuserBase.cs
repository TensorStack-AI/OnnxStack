using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
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
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers
{
    public abstract class DiffuserBase : IDiffuser
    {
        protected readonly IPromptService _promptService;
        protected readonly IOnnxModelService _onnxModelService;

        /// <summary>
        /// Initializes a new instance of the <see cref="DiffuserBase"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="onnxModelService">The onnx model service.</param>
        public DiffuserBase(IOnnxModelService onnxModelService, IPromptService promptService)
        {
            _promptService = promptService;
            _onnxModelService = onnxModelService;
        }


        /// <summary>
        /// Gets the timesteps.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <returns></returns>
        protected abstract IReadOnlyList<int> GetTimesteps(PromptOptions prompt, SchedulerOptions options, IScheduler scheduler);

        /// <summary>
        /// Prepares the latents.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <returns></returns>
        protected abstract DenseTensor<float> PrepareLatents(IModelOptions model, PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps);


        /// <summary>
        /// Rund the stable diffusion loop
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="progress">The progress.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public virtual async Task<DenseTensor<float>> DiffuseAsync(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default)
        {
            // Create random seed if none was set
            schedulerOptions.Seed = schedulerOptions.Seed > 0 ? schedulerOptions.Seed : Random.Shared.Next();

            // Get Scheduler
            using (var scheduler = GetScheduler(promptOptions, schedulerOptions))
            {
                // Process prompts
                var promptEmbeddings = await _promptService.CreatePromptAsync(modelOptions, promptOptions, schedulerOptions);

                // Should we perform classifier free guidance
                var performGuidance = schedulerOptions.GuidanceScale > 1.0f;

                // Get timesteps
                var timesteps = GetTimesteps(promptOptions, schedulerOptions, scheduler);

                // Create latent sample
                var latents = PrepareLatents(modelOptions, promptOptions, schedulerOptions, scheduler, timesteps);

                // Loop though the timesteps
                var step = 0;
                foreach (var timestep in timesteps)
                {
                    step++;
                    cancellationToken.ThrowIfCancellationRequested();

                    // Create input tensor.
                    var inputLatent = performGuidance
                        ? latents.Repeat(2)
                        : latents;
                    var inputTensor = scheduler.ScaleInput(inputLatent, timestep);

                    // Create Input Parameters
                    var inputParameters = CreateUnetInputParams(modelOptions, inputTensor, promptEmbeddings, timestep);

                    // Run Inference
                    using (var inferResult = await _onnxModelService.RunInferenceAsync(modelOptions, OnnxModelType.Unet, inputParameters))
                    {
                        var noisePred = inferResult.FirstElementAs<DenseTensor<float>>();

                        // Perform guidance
                        if (performGuidance)
                            noisePred = PerformGuidance(noisePred, schedulerOptions.GuidanceScale);

                        // Scheduler Step
                        latents = scheduler.Step(noisePred, timestep, latents);
                    }

                    progressCallback?.Invoke(step, timesteps.Count);
                }

                // Decode Latents
                return await DecodeLatents(modelOptions, schedulerOptions, latents);
            }
        }

        /// <summary>
        /// Decodes the latents.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="latents">The latents.</param>
        /// <returns></returns>
        protected async Task<DenseTensor<float>> DecodeLatents(IModelOptions model, SchedulerOptions options, DenseTensor<float> latents)
        {
            // Scale and decode the image latents with vae.
            latents = latents.MultiplyBy(1.0f / model.ScaleFactor);

            var inputNames = _onnxModelService.GetInputNames(model, OnnxModelType.VaeDecoder);
            var inputParameters = CreateInputParameters(NamedOnnxValue.CreateFromTensor(inputNames[0], latents));

            // Run inference.
            using (var inferResult = await _onnxModelService.RunInferenceAsync(model, OnnxModelType.VaeDecoder, inputParameters))
            {
                var resultTensor = inferResult.FirstElementAs<DenseTensor<float>>();
                if (await _onnxModelService.IsEnabledAsync(model, OnnxModelType.SafetyChecker))
                {
                    // Check if image contains NSFW content, 
                    if (!await IsImageSafe(model, options, resultTensor))
                        return resultTensor.CloneEmpty().ToDenseTensor(); //TODO: blank image?, exception?, null?
                }
                return resultTensor.ToDenseTensor();
            }
        }


        /// <summary>
        /// Performs classifier free guidance
        /// </summary>
        /// <param name="noisePredUncond">The noise pred.</param>
        /// <param name="noisePredText">The noise pred text.</param>
        /// <param name="guidanceScale">The guidance scale.</param>
        /// <returns></returns>
        protected DenseTensor<float> PerformGuidance(DenseTensor<float> noisePrediction, float guidanceScale)
        {
            // Split Prompt and Negative Prompt predictions
            var dimensions = noisePrediction.Dimensions.ToArray();
            dimensions[0] /= 2;

            var length = (int)noisePrediction.Length / 2;
            var noisePredCond = new DenseTensor<float>(noisePrediction.Buffer[length..], dimensions);
            var noisePredUncond = new DenseTensor<float>(noisePrediction.Buffer[..length], dimensions);
            return noisePredUncond
                .Add(noisePredCond
                .Subtract(noisePredUncond)
                .MultiplyBy(guidanceScale));
        }


        /// <summary>
        /// Creates the Unet input parameters.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="inputTensor">The input tensor.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        protected virtual IReadOnlyList<NamedOnnxValue> CreateUnetInputParams(IModelOptions model, DenseTensor<float> inputTensor, DenseTensor<float> promptEmbeddings, int timestep)
        {
            var inputNames = _onnxModelService.GetInputNames(model, OnnxModelType.Unet);
            return CreateInputParameters(
                 NamedOnnxValue.CreateFromTensor(inputNames[0], inputTensor),
                 NamedOnnxValue.CreateFromTensor(inputNames[1], new DenseTensor<long>(new long[] { timestep }, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor(inputNames[2], promptEmbeddings));
        }


        /// <summary>
        /// Determines whether the specified result image is not NSFW.
        /// </summary>
        /// <param name="resultImage">The result image.</param>
        /// <param name="config">The configuration.</param>
        /// <returns>
        ///   <c>true</c> if the specified result image is safe; otherwise, <c>false</c>.
        /// </returns>
        protected async Task<bool> IsImageSafe(IModelOptions model, SchedulerOptions options, DenseTensor<float> resultImage)
        {
            //clip input
            var inputTensor = ClipImageFeatureExtractor(options, resultImage);

            //images input
            var inputNames = _onnxModelService.GetInputNames(model, OnnxModelType.SafetyChecker);
            var inputImagesTensor = inputTensor.ReorderTensor(new[] { 1, 224, 224, 3 });
            var inputParameters = CreateInputParameters(
                NamedOnnxValue.CreateFromTensor(inputNames[0], inputTensor),
                NamedOnnxValue.CreateFromTensor(inputNames[1], inputImagesTensor));

            // Run session and send the input data in to get inference output. 
            using (var inferResult = await _onnxModelService.RunInferenceAsync(model, OnnxModelType.SafetyChecker, inputParameters))
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
        protected static DenseTensor<float> ClipImageFeatureExtractor(SchedulerOptions options, DenseTensor<float> imageTensor)
        {
            //convert tensor result to image
            using (var image = imageTensor.ToImage())
            {
                // Resize image
                ImageHelpers.Resize(image, new[] { 1, 3, 224, 224 });

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
        protected static IScheduler GetScheduler(PromptOptions prompt, SchedulerOptions options)
        {
            return prompt.SchedulerType switch
            {
                SchedulerType.LMS => new LMSScheduler(options),
                SchedulerType.EulerAncestral => new EulerAncestralScheduler(options),
                SchedulerType.DDPM => new DDPMScheduler(options),
                _ => default
            };
        }

        /// <summary>
        /// Helper for creating the input parameters.
        /// </summary>
        /// <param name="parameters">The parameters.</param>
        /// <returns></returns>
        protected static IReadOnlyList<NamedOnnxValue> CreateInputParameters(params NamedOnnxValue[] parameters)
        {
            return parameters.ToList();
        }
    }
}
