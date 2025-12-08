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

namespace OnnxStack.StableDiffusion.AMD.StableDiffusion3
{
    public class AMDNPUImageDiffuser : AMDNPUDiffuser
    {
        public AMDNPUImageDiffuser(UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, ILogger logger = null)
            : base(unet, vaeDecoder, vaeEncoder, logger)
        {
        }


        public override DiffuserType DiffuserType => DiffuserType.ImageToImage;


        protected override IReadOnlyList<int> GetTimesteps(SchedulerOptions options, IScheduler scheduler)
        {
            if (!options.Timesteps.IsNullOrEmpty())
                return options.Timesteps;

            return scheduler.Timesteps
                .Skip(options.GetStrengthScaledStartingStep())
                .ToList();
        }


        protected override async Task<DenseTensor<float>> PrepareLatentsAsync(GenerateOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger.LogBegin();
            var imageTensor = await options.InputImage.GetImageTensorAsync(options.SchedulerOptions.Height, options.SchedulerOptions.Width);
            var outputDimension = new[] { 1, 16, imageTensor.Dimensions[2] / 8, imageTensor.Dimensions[3] / 8 };
            var metadata = await _vaeEncoder.LoadAsync(cancellationToken: cancellationToken);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
            {
                inferenceParameters.AddInputTensor(imageTensor);
                inferenceParameters.AddOutputBuffer(outputDimension);
                var results = await _vaeEncoder.RunInferenceAsync(inferenceParameters);
                using (var result = results.First())
                {
                    var scaledSample = result
                        .ToDenseTensor()
                        .Subtract(ShiftFactor)
                        .MultiplyBy(_vaeEncoder.ScaleFactor);

                    if (options.IsLowMemoryDecoderEnabled)
                        await _vaeEncoder.UnloadAsync();

                    _logger?.LogEnd(LogLevel.Debug, "VaeEncoder", timestamp);
                    return scheduler.AddNoise(scaledSample, scheduler.CreateRandomSample(scaledSample.Dimensions), timesteps);
                }
            }
        }

    }
}
