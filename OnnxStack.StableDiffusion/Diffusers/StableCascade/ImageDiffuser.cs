using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers.StableCascade
{
    public sealed class ImageDiffuser : StableCascadeDiffuser
    {

        /// <summary>
        /// Initializes a new instance of the <see cref="ImageDiffuser"/> class.
        /// </summary>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public ImageDiffuser(UNetConditionModel priorUnet, UNetConditionModel decoderUnet, AutoEncoderModel decoderVqgan, AutoEncoderModel imageEncoder, ILogger logger = default)
            : base(priorUnet, decoderUnet, decoderVqgan, imageEncoder, logger) { }


        /// <summary>
        /// Gets the type of the diffuser.
        /// </summary>
        public override DiffuserType DiffuserType => DiffuserType.ImageToImage;


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


        /// <summary>
        /// Encodes the image.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="performGuidance">if set to <c>true</c> [perform guidance].</param>
        /// <returns></returns>
        protected override async Task<DenseTensor<float>> EncodeImageAsync(GenerateOptions options, bool performGuidance, CancellationToken cancellationToken = default)
        {
            var metadata = await _vaeEncoder.LoadAsync(cancellationToken: cancellationToken);
            var imageTensor = options.InputImage.GetClipImageFeatureTensor();
            using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
            {
                inferenceParameters.AddInputTensor(imageTensor);
                inferenceParameters.AddOutputBuffer(new[] { 1, ClipImageChannels });

                var results = await _vaeEncoder.RunInferenceAsync(inferenceParameters);
                using (var result = results.First())
                {
                    // Unload if required
                    if (options.IsLowMemoryEncoderEnabled)
                        await _vaeEncoder.UnloadAsync();

                    var image_embeds = result.ToDenseTensor(new[] { 1, 1, ClipImageChannels });
                    if (performGuidance)
                        return new DenseTensor<float>(image_embeds.Dimensions).Concatenate(image_embeds);

                    return image_embeds;
                }
            }
        }
    }
}
