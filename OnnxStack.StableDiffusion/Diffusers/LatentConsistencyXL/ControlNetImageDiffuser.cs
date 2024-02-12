using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Models;
using SixLabors.ImageSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers.LatentConsistencyXL
{
    public sealed class ControlNetImageDiffuser : ControlNetDiffuser
    {

        /// <summary>
        /// Initializes a new instance of the <see cref="ControlNetImageDiffuser"/> class.
        /// </summary>
        /// <param name="controlNet">The control net.</param>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public ControlNetImageDiffuser(ControlNetModel controlNet, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, MemoryModeType memoryMode, ILogger logger = default)
            : base(controlNet, unet, vaeDecoder, vaeEncoder, memoryMode, logger) { }


        /// <summary>
        /// Gets the type of the diffuser.
        /// </summary>
        public override DiffuserType DiffuserType => DiffuserType.ControlNetImage;


        /// <summary>
        /// Gets the timesteps.
        /// </summary>
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
        protected override async Task<DenseTensor<float>> PrepareLatentsAsync(PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps)
        {
            var imageTensor = await prompt.InputImage.GetImageTensorAsync(options.Height, options.Width);
            var outputDimension = options.GetScaledDimension();
            var metadata = await _vaeEncoder.GetMetadataAsync();
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(imageTensor);
                inferenceParameters.AddOutputBuffer(outputDimension);

                var results = await _vaeEncoder.RunInferenceAsync(inferenceParameters);
                using (var result = results.First())
                {
                    // Unload if required
                    if (_memoryMode == MemoryModeType.Minimum)
                        await _vaeEncoder.UnloadAsync();

                    var outputResult = result.ToDenseTensor();
                    var scaledSample = outputResult.MultiplyBy(_vaeEncoder.ScaleFactor);
                    return scheduler.AddNoise(scaledSample, scheduler.CreateRandomSample(scaledSample.Dimensions), timesteps);
                }
            }
        }

    }
}
