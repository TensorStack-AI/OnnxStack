using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Image;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using SixLabors.ImageSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers.Locomotion
{
    public sealed class ControlNetVideoDiffuser : ControlNetDiffuser
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ControlNetVideoDiffuser"/> class.
        /// </summary>
        /// <param name="controlNet">The unet.</param>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="memoryMode">The memory mode.</param>
        /// <param name="logger">The logger.</param>
        public ControlNetVideoDiffuser(ControlNetModel controlNet, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, FlowEstimationModel flowEstimation, ResampleModel resampler, ILogger logger = default)
            : base(controlNet, unet, vaeDecoder, vaeEncoder, flowEstimation, resampler, logger) { }


        /// <summary>
        /// Gets the type of the diffuser.
        /// </summary>
        public override DiffuserType DiffuserType => DiffuserType.ControlNetVideo;


        /// <summary>
        /// Gets the timesteps.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <returns></returns>
        protected override IReadOnlyList<int> GetTimesteps(SchedulerOptions options, IScheduler scheduler)
        {
            if (!options.Timesteps.IsNullOrEmpty())
                return options.Timesteps;

            return scheduler.Timesteps
                .Skip(options.GetStrengthScaledStartingStep())
                .ToList();
        }


        /// <summary>
        /// Prepares the input latents.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;DenseTensor`1&gt; representing the asynchronous operation.</returns>
        protected override async Task<DenseTensor<float>> PrepareLatentsAsync(GenerateOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps, CancellationToken cancellationToken = default)
        {
            if (options.SchedulerOptions.Strength < 1f)
                return await PrepareVideoLatentsAsync(options, scheduler, timesteps, cancellationToken);

            var frameCount = (int)Math.Ceiling(options.FrameCount / (double)_unet.ContextSize) * _unet.ContextSize;
            return await PrepareNoiseLatentsAsync(options, scheduler, frameCount);
        }


        /// <summary>
        /// Prepares the control latents.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns>DenseTensor&lt;System.Single&gt;.</returns>
        protected override async Task<DenseTensor<float>> PrepareControlLatents(GenerateOptions options)
        {
            var controlLatents = new List<DenseTensor<float>>();
            foreach (var frame in GetContextFrames(options.InputContolVideo.Frames, _unet.ContextSize))
            {
                controlLatents.Add(await frame.GetImageTensorAsync(options.SchedulerOptions.Height, options.SchedulerOptions.Width, ImageNormalizeType.ZeroToOne));
            }

            var contronNetInputTensor = controlLatents.Join();
            if (_controlNet.InvertInput)
                InvertInputTensor(contronNetInputTensor);

            return contronNetInputTensor;
        }
    }
}
