using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using SixLabors.ImageSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers.StableDiffusionXL
{
    public sealed class ControlNetImageDiffuser : ControlNetDiffuser
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ControlNetImageDiffuser"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="onnxModelService">The onnx model service.</param>
        public ControlNetImageDiffuser(IOnnxModelService onnxModelService, IPromptService promptService, IControlNetImageService controlNetImageService, ILogger<ControlNetDiffuser> logger)
            : base(onnxModelService, promptService, controlNetImageService, logger)
        {
        }


        /// <summary>
        /// Gets the type of the diffuser.
        /// </summary>
        public override DiffuserType DiffuserType => DiffuserType.ControlNetImage;


        /// <summary>
        /// Gets the timesteps.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <returns></returns>
        protected override IReadOnlyList<int> GetTimesteps(SchedulerOptions options, IScheduler scheduler)
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
        protected override async Task<DenseTensor<float>> PrepareLatentsAsync(ModelOptions model, PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps)
        {
            var imageTensor = await prompt.InputImage.ToDenseTensorAsync(options.Height, options.Width);

            //TODO: Model Config, Channels
            var outputDimension = options.GetScaledDimension();
            var metadata = _onnxModelService.GetModelMetadata(model.BaseModel, OnnxModelType.VaeEncoder);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(imageTensor);
                inferenceParameters.AddOutputBuffer(outputDimension);

                var results = await _onnxModelService.RunInferenceAsync(model.BaseModel, OnnxModelType.VaeEncoder, inferenceParameters);
                using (var result = results.First())
                {
                    var outputResult = result.ToDenseTensor();
                    var scaledSample = outputResult.MultiplyBy(model.ScaleFactor);
                    return scheduler.AddNoise(scaledSample, scheduler.CreateRandomSample(scaledSample.Dimensions), timesteps);
                }
            }
        }

    }
}
