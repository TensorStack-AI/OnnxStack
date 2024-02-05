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
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers
{
    public abstract class DiffuserBase : IDiffuser
    {
        protected readonly ILogger _logger;
        protected readonly UNetConditionModel _unet;
        protected readonly AutoEncoderModel _vaeDecoder;
        protected readonly AutoEncoderModel _vaeEncoder;
        protected readonly MemoryModeType _memoryMode;

        /// <summary>
        /// Initializes a new instance of the <see cref="DiffuserBase"/> class.
        /// </summary>
        /// <param name="modelConfig">The model configuration.</param>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public DiffuserBase(UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, MemoryModeType memoryMode, ILogger logger = default)
        {
            _logger = logger;
            _unet = unet;
            _vaeDecoder = vaeDecoder;
            _vaeEncoder = vaeEncoder;
            _memoryMode = memoryMode;
        }

        /// <summary>
        /// Gets the type of the diffuser.
        /// </summary>
        public abstract DiffuserType DiffuserType { get; }

        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public abstract DiffuserPipelineType PipelineType { get; }

        /// <summary>
        /// Gets the scheduler.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns></returns>
        protected abstract IScheduler GetScheduler(SchedulerOptions options);

        /// <summary>
        /// Gets the timesteps.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <returns></returns>
        protected abstract IReadOnlyList<int> GetTimesteps(SchedulerOptions options, IScheduler scheduler);


        /// <summary>
        /// Prepares the input latents.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <returns></returns>
        protected abstract Task<DenseTensor<float>> PrepareLatentsAsync(PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps);


        /// <summary>
        /// Runs the diffusion process
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <param name="performGuidance">if set to <c>true</c> [perform guidance].</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public abstract Task<DenseTensor<float>> DiffuseAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions, PromptEmbeddingsResult promptEmbeddings, bool performGuidance, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);



        /// <summary>
        /// Performs classifier free guidance
        /// </summary>
        /// <param name="noisePredUncond">The noise pred.</param>
        /// <param name="noisePredText">The noise pred text.</param>
        /// <param name="guidanceScale">The guidance scale.</param>
        /// <returns></returns>
        protected virtual DenseTensor<float> PerformGuidance(DenseTensor<float> noisePrediction, float guidanceScale)
        {
            // Split Prompt and Negative Prompt predictions
            var dimensions = noisePrediction.Dimensions.ToArray();
            dimensions[0] /= 2;

            var length = (int)noisePrediction.Length / 2;
            var noisePredCond = new DenseTensor<float>(noisePrediction.Buffer[length..], dimensions);
            var noisePredUncond = new DenseTensor<float>(noisePrediction.Buffer[..length], dimensions);
            return noisePredUncond
                .Add(noisePredCond
                .Subtract(noisePredUncond)
                .MultiplyBy(guidanceScale));
        }


        /// <summary>
        /// Decodes the latents.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="latents">The latents.</param>
        /// <returns></returns>
        protected virtual async Task<DenseTensor<float>> DecodeLatentsAsync(PromptOptions prompt, SchedulerOptions options, DenseTensor<float> latents)
        {
            var timestamp = _logger.LogBegin();

            // Scale and decode the image latents with vae.
            latents = latents.MultiplyBy(1.0f / _vaeDecoder.ScaleFactor);

            var outputDim = new[] { 1, 3, options.Height, options.Width };
            var metadata = await _vaeDecoder.GetMetadataAsync();
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(latents);
                inferenceParameters.AddOutputBuffer(outputDim);

                var results = await _vaeDecoder.RunInferenceAsync(inferenceParameters);
                using (var imageResult = results.First())
                {
                    // Unload if required
                    if (_memoryMode == MemoryModeType.Minimum)
                        await _vaeDecoder.UnloadAsync();

                    _logger?.LogEnd("Latents decoded", timestamp);
                    return imageResult.ToDenseTensor();
                }
            }

        }


        /// <summary>
        /// Creates the timestep tensor.
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        protected static DenseTensor<float> CreateTimestepTensor(int timestep)
        {
            return TensorHelper.CreateTensor(new float[] { timestep }, new int[] { 1 });
        }


        /// <summary>
        /// Reports the progress.
        /// </summary>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="progress">The progress.</param>
        /// <param name="progressMax">The progress maximum.</param>
        /// <param name="output">The output.</param>
        protected void ReportProgress(Action<DiffusionProgress> progressCallback, int progress, int progressMax, DenseTensor<float> progressTensor)
        {
            progressCallback?.Invoke(new DiffusionProgress
            {
                StepMax = progressMax,
                StepValue = progress,
                StepTensor = progressTensor
            });
        }
    }
}
