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

namespace OnnxStack.StableDiffusion.Diffusers.StableDiffusion3
{
    public class ControlNetImageDiffuser : ControlNetDiffuser
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ControlNetImageDiffuser"/> class.
        /// </summary>
        /// <param name="controlNet">The control net.</param>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public ControlNetImageDiffuser(ControlNetModel controlNet, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, ILogger logger = default)
            : base(controlNet, unet, vaeDecoder, vaeEncoder, logger) { }


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

            try
            {
                if (options.IsAutoEncoderTileEnabled)
                    return await EncodeLatentsTilesAsync(imageTensor, options, scheduler, timesteps, cancellationToken);

                return await EncodeLatentsAsync(imageTensor, options, scheduler, timesteps, cancellationToken);
            }
            finally
            {
                if (options.IsLowMemoryDecoderEnabled)
                    await _vaeEncoder.UnloadAsync();

                _logger?.LogEnd(LogLevel.Debug, "VaeEncoder", timestamp);
            }
        }


        /// <summary>
        /// Encode the input latents
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;DenseTensor`1&gt; representing the asynchronous operation.</returns>
        private async Task<DenseTensor<float>> EncodeLatentsAsync(DenseTensor<float> imageTensor, GenerateOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps, CancellationToken cancellationToken = default)
        {
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
                    return scheduler.AddNoise(scaledSample, scheduler.CreateRandomSample(scaledSample.Dimensions), timesteps);
                }
            }
        }


        /// <summary>
        /// Encode the input latents as tiles
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;DenseTensor`1&gt; representing the asynchronous operation.</returns>
        private async Task<DenseTensor<float>> EncodeLatentsTilesAsync(DenseTensor<float> imageTensor, GenerateOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps, CancellationToken cancellationToken = default)
        {
            var tileSize = 512;
            var scaleFactor = 8;
            var tileOverlap = 16;
            var width = imageTensor.Dimensions[3];
            var height = imageTensor.Dimensions[2];
            var tileMode = options.AutoEncoderTileMode;
            if (width <= (tileSize + tileOverlap) || height <= (tileSize + tileOverlap))
                return await DecodeLatentsAsync(imageTensor, options, cancellationToken);

            var inputTiles = new ImageTiles(imageTensor, tileMode, tileOverlap);
            var outputTiles = new ImageTiles
            (
                inputTiles.Width / scaleFactor,
                inputTiles.Height / scaleFactor,
                tileMode,
                inputTiles.Overlap / scaleFactor,
                await EncodeLatentsAsync(inputTiles.Tile1, options, scheduler, timesteps, cancellationToken),
                await EncodeLatentsAsync(inputTiles.Tile2, options, scheduler, timesteps, cancellationToken),
                await EncodeLatentsAsync(inputTiles.Tile3, options, scheduler, timesteps, cancellationToken),
                await EncodeLatentsAsync(inputTiles.Tile4, options, scheduler, timesteps, cancellationToken)
            );
            return outputTiles.JoinTiles();
        }
    }
}
