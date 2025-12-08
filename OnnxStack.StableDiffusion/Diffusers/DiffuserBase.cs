using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
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

        /// <summary>
        /// Initializes a new instance of the <see cref="DiffuserBase"/> class.
        /// </summary>
        /// <param name="modelConfig">The model configuration.</param>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public DiffuserBase(UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, ILogger logger = default)
        {
            _logger = logger;
            _unet = unet;
            _vaeDecoder = vaeDecoder;
            _vaeEncoder = vaeEncoder;
        }

        /// <summary>
        /// Gets the type of the diffuser.
        /// </summary>
        public abstract DiffuserType DiffuserType { get; }

        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public abstract PipelineType PipelineType { get; }

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
        protected abstract Task<DenseTensor<float>> PrepareLatentsAsync(GenerateOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps, CancellationToken cancellationToken = default);


        /// <summary>
        /// Runs the diffusion process
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public abstract Task<DenseTensor<float>> DiffuseAsync(DiffuseOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Runs the video Diffusion process
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public virtual async IAsyncEnumerable<DenseTensor<float>> DiffuseVideoAsync(DiffuseOptions options, IProgress<DiffusionProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var frameIndex = 0;
            var previousFrame = default(OnnxImage);
            var generateOptions = options.GenerateOptions;
            var videoFrames = generateOptions.InputVideo.Frames;
            foreach (var videoFrame in videoFrames)
            {
                var frameTimestamp = Stopwatch.GetTimestamp();
                if (generateOptions.Diffuser == DiffuserType.ControlNet || generateOptions.Diffuser == DiffuserType.ControlNetImage)
                {
                    // ControlNetImage uses frame as input image
                    if (generateOptions.Diffuser == DiffuserType.ControlNetImage)
                        generateOptions.InputImage = BlendVideoFrames(generateOptions, videoFrame, previousFrame);

                    generateOptions.InputContolImage = generateOptions.InputContolVideo?.GetFrame(frameIndex);
                }
                else
                {
                    generateOptions.InputImage = BlendVideoFrames(generateOptions, videoFrame, previousFrame);
                }

                var imageOptions = new DiffuseOptions(generateOptions, options.PromptEmbeddings);
                var frameResultTensor = await DiffuseAsync(imageOptions, progressCallback, cancellationToken);
                previousFrame = new OnnxImage(frameResultTensor);

                // Frame Progress
                ReportFrameProgress(progressCallback, ++frameIndex, videoFrames.Count, frameResultTensor, frameTimestamp);

                yield return frameResultTensor;
            }
        }

        protected virtual bool ShouldPerformGuidance(SchedulerOptions schedulerOptions)
        {
            return schedulerOptions.GuidanceScale > 1f;
        }

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
            var noisePredCond = noisePrediction.Buffer.Span[length..];
            var noisePredUncond = noisePrediction.Buffer.Span[..length];
            noisePredUncond.Lerp(noisePredCond, guidanceScale);
            return new DenseTensor<float>(noisePredUncond.ToArray(), dimensions);
        }


        /// <summary>
        /// Decodes the latents.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="latents">The latents.</param>
        /// <returns></returns>
        protected virtual async Task<DenseTensor<float>> DecodeLatentsAsync(GenerateOptions options, DenseTensor<float> latents, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger.LogBegin();
            latents = latents.MultiplyBy(1.0f / _vaeDecoder.ScaleFactor);
            var outputDim = new[] { 1, 3, options.SchedulerOptions.Height, options.SchedulerOptions.Width };
            var metadata = await _vaeDecoder.LoadAsync(cancellationToken: cancellationToken);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
            {
                inferenceParameters.AddInputTensor(latents);
                inferenceParameters.AddOutputBuffer(outputDim);

                var results = await _vaeDecoder.RunInferenceAsync(inferenceParameters);
                using (var imageResult = results.First())
                {
                    if (options.IsLowMemoryDecoderEnabled)
                        await _vaeDecoder.UnloadAsync();

                    _logger?.LogEnd(LogLevel.Debug, "VaeDecoder", timestamp);
                    return imageResult.ToDenseTensor();
                }
            }
        }


        /// <summary>
        /// Creates the timestep tensor.
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        protected virtual DenseTensor<float> CreateTimestepTensor(int timestep)
        {
            return new DenseTensor<float>(new float[] { timestep }, new int[] { 1 });
        }


        /// <summary>
        /// Reports the progress.
        /// </summary>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="message">The message.</param>
        /// <param name="progress">The progress.</param>
        /// <param name="progressMax">The progress maximum.</param>
        /// <param name="elapsed">The elapsed.</param>
        /// <param name="progressTensor">The progress tensor.</param>
        protected void ReportProgress(IProgress<DiffusionProgress> progressCallback, string message, int progress, int progressMax, long elapsed = 0, DenseTensor<float> progressTensor = default)
        {
            progressCallback?.Report(new DiffusionProgress
            {
                StepMax = progressMax,
                StepValue = progress,
                StepTensor = progressTensor.CloneTensor(),
                Message = $"{message}: {progress:D2}/{progressMax:D2}",
                Elapsed = elapsed > 0 ? Stopwatch.GetElapsedTime(elapsed).TotalMilliseconds : 0.0
            });
        }

        protected void ReportFrameProgress(IProgress<DiffusionProgress> progressCallback, int progress, int progressMax, DenseTensor<float> progressTensor, long elapsed)
        {
            progressCallback?.Report(new DiffusionProgress(elapsed)
            {
                BatchMax = progressMax,
                BatchValue = progress,
                BatchTensor = progressTensor
            });
        }


        protected OnnxImage BlendVideoFrames(GenerateOptions options, OnnxImage frame1, OnnxImage frame2)
        {
            if (!options.IsFrameBlendEnabled || frame2 is null)
                return frame1;

            return frame1.Merge(frame2, options.PreviousFrameStrength, options.FrameBlendingMode, options.FrameStrength);
        }
    }
}
