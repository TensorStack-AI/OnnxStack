using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers.StableDiffusionXL;
using OnnxStack.StableDiffusion.Models;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.AMD.StableDiffusionXL
{
    public class AMDTextDiffuser : TextDiffuser
    {
        public AMDTextDiffuser(UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, ILogger logger = null)
            : base(unet, vaeDecoder, vaeEncoder, logger)
        {
        }


        public override Task<DenseTensor<float>> DiffuseAsync(DiffuseOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            // TODO: Offload inference to c++ engine and return tensor result
            return base.DiffuseAsync(options, progressCallback, cancellationToken);
        }
    }
}
