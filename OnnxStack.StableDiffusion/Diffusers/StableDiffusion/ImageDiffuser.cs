using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Config;
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
using OnnxStack.Core;

namespace OnnxStack.StableDiffusion.Diffusers.StableDiffusion
{
    public sealed class ImageDiffuser : StableDiffusionDiffuser
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ImageDiffuser"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="onnxModelService">The onnx model service.</param>
        public ImageDiffuser(IOnnxModelService onnxModelService, IPromptService promptService, ILogger<StableDiffusionDiffuser> logger)
            : base(onnxModelService, promptService, logger)
        {
        }


        /// <summary>
        /// Gets the type of the diffuser.
        /// </summary>
        public override DiffuserType DiffuserType => DiffuserType.ImageToImage;


        /// <summary>
        /// Gets the timesteps.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <returns></returns>
        protected override IReadOnlyList<int> GetTimesteps(SchedulerOptions options, IScheduler scheduler)
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
        protected override async Task<DenseTensor<float>> PrepareLatents(IModelOptions model, PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps)
        {
            var imageTensor = prompt.InputImage.ToDenseTensor(new[] { 1, 3, options.Height, options.Width });
            var inputNames = _onnxModelService.GetInputNames(model, OnnxModelType.VaeEncoder);
            var outputNames = _onnxModelService.GetOutputNames(model, OnnxModelType.VaeEncoder);

            //TODO: Model Config, Channels
            var outputDim = options.GetScaledDimension();
            var outputBuffer = new DenseTensor<float>(outputDim);
            using (var inputTensorValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, imageTensor.Buffer, imageTensor.Dimensions.ToLong()))
            using (var outputTensorValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, outputBuffer.Buffer, outputDim.ToLong()))
            {
                var inputs = new Dictionary<string, OrtValue> { { inputNames[0], inputTensorValue } };
                var outputs = new Dictionary<string, OrtValue> { { outputNames[0], outputTensorValue } };
                var results = await _onnxModelService.RunInferenceAsync(model, OnnxModelType.VaeEncoder, inputs, outputs);
                using (var result = results.First())
                {
                    var scaledSample = outputBuffer
                       .Add(scheduler.CreateRandomSample(outputBuffer.Dimensions, options.InitialNoiseLevel))
                       .MultiplyBy(model.ScaleFactor);

                    var noisySample = scheduler.AddNoise(scaledSample, scheduler.CreateRandomSample(scaledSample.Dimensions), timesteps);
                    if (prompt.BatchCount > 1)
                        return noisySample.Repeat(prompt.BatchCount);

                    return noisySample;
                }
            }
        }

    }
}
