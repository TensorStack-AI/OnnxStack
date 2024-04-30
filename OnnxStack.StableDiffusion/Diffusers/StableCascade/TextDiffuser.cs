using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers.StableCascade
{
    public sealed class TextDiffuser : StableCascadeDiffuser
    {

        /// <summary>
        /// Initializes a new instance of the <see cref="TextDiffuser"/> class.
        /// </summary>
        /// <param name="priorUnet">The prior unet.</param>
        /// <param name="decoderUnet">The decoder unet.</param>
        /// <param name="decoderVqgan">The decoder vqgan.</param>
        /// <param name="imageEncoder">The image encoder.</param>
        /// <param name="memoryMode">The memory mode.</param>
        /// <param name="logger">The logger.</param>
        public TextDiffuser(UNetConditionModel priorUnet, UNetConditionModel decoderUnet, AutoEncoderModel decoderVqgan, MemoryModeType memoryMode, ILogger logger = default)
            : base(priorUnet, decoderUnet, decoderVqgan, default, memoryMode, logger) { }


        /// <summary>
        /// Gets the type of the diffuser.
        /// </summary>
        public override DiffuserType DiffuserType => DiffuserType.TextToImage;


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

            return scheduler.Timesteps;
        }


        protected override Task<DenseTensor<float>> PrepareLatentsAsync(PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps)
        {
            var latents = scheduler.CreateRandomSample(new[]
            {
               1, 16,
               (int)Math.Ceiling(options.Height / ResolutionMultiple),
               (int)Math.Ceiling(options.Width / ResolutionMultiple)
           }, scheduler.InitNoiseSigma);
            return Task.FromResult(latents);
        }


        protected override Task<DenseTensor<float>> PrepareDecoderLatentsAsync(PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps, DenseTensor<float> priorLatents)
        {
            var latents = scheduler.CreateRandomSample(new[]
            {
                1, 4,
                (int)(priorLatents.Dimensions[2] * LatentDimScale),
                (int)(priorLatents.Dimensions[3] * LatentDimScale)
            }, scheduler.InitNoiseSigma);
            return Task.FromResult(latents);
        }
    }
}
