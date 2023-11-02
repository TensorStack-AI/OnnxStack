using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Config;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Helpers;
using SixLabors.ImageSharp;
using System;
using System.Collections.Generic;
using System.Linq;


namespace OnnxStack.StableDiffusion.Diffusers.LatentConsistency
{
    public sealed class ImageDiffuser : TextDiffuser
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ImageDiffuser"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="onnxModelService">The onnx model service.</param>
        public ImageDiffuser(IOnnxModelService onnxModelService, IPromptService promptService)
            : base(onnxModelService, promptService)
        {
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
        protected override DenseTensor<float> PrepareLatents(IModelOptions model, PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps)
        {
            // Image input, decode, add noise, return as latent 0
            var imageTensor = prompt.InputImage.ToDenseTensor(new[] { 1, 3, options.Height, options.Width });
            var inputNames = _onnxModelService.GetInputNames(model, OnnxModelType.VaeEncoder);
            var inputParameters = CreateInputParameters(NamedOnnxValue.CreateFromTensor(inputNames[0], imageTensor));
            using (var inferResult = _onnxModelService.RunInference(model, OnnxModelType.VaeEncoder, inputParameters))
            {
                var sample = inferResult.FirstElementAs<DenseTensor<float>>();
                var scaledSample = sample
                    .Add(scheduler.CreateRandomSample(sample.Dimensions, options.InitialNoiseLevel))
                    .MultiplyBy(model.ScaleFactor);

                var noisySample = scheduler.AddNoise(scaledSample, scheduler.CreateRandomSample(scaledSample.Dimensions), timesteps);
                if (prompt.BatchCount > 1)
                    return noisySample.Repeat(prompt.BatchCount);

                return noisySample;
            }
        }

    }
}
