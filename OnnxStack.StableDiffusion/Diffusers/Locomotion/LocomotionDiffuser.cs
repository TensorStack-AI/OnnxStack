using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.StableDiffusion.Schedulers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers.Locomotion
{
    public abstract class LocomotionDiffuser : DiffuserBase
    {
        private readonly ResampleModel _resampler;
        private readonly FlowEstimationModel _flowEstimation;


        /// <summary>
        /// Initializes a new instance of the <see cref="LocomotionDiffuser"/> class.
        /// </summary>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="memoryMode"></param>
        /// <param name="logger">The logger.</param>
        public LocomotionDiffuser(UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, FlowEstimationModel flowEstimation, ResampleModel resampler, ILogger logger = default)
            : base(unet, vaeDecoder, vaeEncoder, logger)
        {
            _resampler = resampler;
            _flowEstimation = flowEstimation;
        }


        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override PipelineType PipelineType => PipelineType.Locomotion;


        /// <summary>
        /// Runs the diffusion process
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async Task<DenseTensor<float>> DiffuseAsync(DiffuseOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var videoFrames = await DiffuseInternalAsync(options, progressCallback, cancellationToken).ToListAsync(cancellationToken: cancellationToken);
            return videoFrames.Join();
        }


        /// <summary>
        /// Runs the video Diffusion process
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override IAsyncEnumerable<DenseTensor<float>> DiffuseVideoAsync(DiffuseOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            return DiffuseInternalAsync(options, progressCallback, cancellationToken);
        }


        /// <summary>
        /// Runs the Locomotion Diffusion process
        /// </summary>
        /// <param name="generateOptions">The generate options.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        protected virtual async IAsyncEnumerable<DenseTensor<float>> DiffuseInternalAsync(DiffuseOptions options, IProgress<DiffusionProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var contextSize = _unet.ContextSize;
            var generateOptions = options.GenerateOptions;
            var promptEmbeddings = options.PromptEmbeddings;
            var schedulerOptions = generateOptions.SchedulerOptions;
            var promptEmbeddingsCond = promptEmbeddings.PromptEmbeds;
            var promptEmbeddingsUncond = promptEmbeddings.NegativePromptEmbeds;

            var optimizations = GetOptimizations(generateOptions, promptEmbeddings, progressCallback);
            using (var scheduler = GetScheduler(schedulerOptions))
            {
                // Get timesteps
                var timesteps = GetTimesteps(schedulerOptions, scheduler);

                // Get Model metadata
                var metadata = await _unet.LoadAsync(optimizations, cancellationToken);

                // Create latent sample
                progressCallback.Notify("Prepare Input Frames...");
                var latents = await PrepareLatentsAsync(generateOptions, scheduler, timesteps, cancellationToken);

                // Setup motion/context parameters
                var totalFrames = latents.Dimensions[2];
                var contextCount = (totalFrames / contextSize);
                var contextStride = generateOptions.MotionStrides;
                var contextOverlap = generateOptions.MotionContextOverlap;
                var contextWindowCount = GetContextWindows(0, totalFrames, contextSize, contextStride, contextOverlap).Count;

                var step = 0;
                var contextStep = 0;
                var contextSteps = timesteps.Count * contextWindowCount;
                ReportProgress(progressCallback, "Step", 0, contextSteps, 0);
                foreach (var timestep in timesteps)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    var performGuidance = ShouldPerformGuidance(schedulerOptions);
                    var timestepTensor = CreateTimestepTensor(timestep);
                    var noisePredCond = new DenseTensor<float>(latents.Dimensions);
                    var noisePredUncond = new DenseTensor<float>(latents.Dimensions);

                    // Loop though the context windows
                    var contextWindows = GetContextWindows(contextStep, totalFrames, contextSize, contextStride, contextOverlap);
                    foreach (var contextWindow in contextWindows)
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        var stepTime = Stopwatch.GetTimestamp();
                        var contextLatents = GetContextWindow(latents, contextWindow);
                        var inputTensor = scheduler.ScaleInput(contextLatents, timestep);
                        var contextPromptCond = GetPromptContextWindow(promptEmbeddingsCond, contextWindow);
                        var contextPromptUncond = GetPromptContextWindow(promptEmbeddingsUncond, contextWindow);

                        using (var unetCondParams = new OnnxInferenceParameters(metadata, cancellationToken))
                        using (var unetUncondParams = new OnnxInferenceParameters(metadata, cancellationToken))
                        {
                            unetCondParams.AddInputTensor(inputTensor);
                            unetCondParams.AddInputTensor(timestepTensor);
                            unetCondParams.AddInputTensor(contextPromptCond);
                            unetCondParams.AddOutputBuffer(inputTensor.Dimensions);

                            var unetUncondResults = default(IReadOnlyCollection<OrtValue>);
                            if (performGuidance)
                            {
                                unetUncondParams.AddInputTensor(inputTensor);
                                unetUncondParams.AddInputTensor(timestepTensor);
                                unetUncondParams.AddInputTensor(contextPromptUncond);
                                unetUncondParams.AddOutputBuffer(inputTensor.Dimensions);
                                unetUncondResults = await _unet.RunInferenceAsync(unetUncondParams);
                            }

                            var unetCondResults = await _unet.RunInferenceAsync(unetCondParams);
                            using (var unetCondResult = unetCondResults.First())
                            using (var unetUncondResult = unetUncondResults?.FirstOrDefault())
                            {
                                UpdateContextWindow(noisePredCond, unetCondResult.ToDenseTensor(), contextWindow);
                                if (performGuidance)
                                    UpdateContextWindow(noisePredUncond, unetUncondResult.ToDenseTensor(), contextWindow);
                            }
                        }

                        step++;
                        ReportProgress(progressCallback, "Step", step, contextSteps, stepTime);
                        _logger?.LogEnd(LogLevel.Debug, $"Step {step}/{contextSteps}", stepTime);
                    }

                    // PerformGuidance
                    if (performGuidance)
                        noisePredCond = PerformGuidance(noisePredCond, noisePredUncond, schedulerOptions.GuidanceScale);

                    // Scheduler Step
                    latents = scheduler.Step(noisePredCond, timestep, latents, generateOptions.MotionNoiseContext).Result;
                    contextStep++;
                }

                // Unload if required
                if (generateOptions.IsLowMemoryComputeEnabled)
                    await _unet.UnloadAsync();

                // Decode Latents
                progressCallback.Notify("Generating Frames...");
                await foreach (var decodedFrame in DecodeVideoLatentsAsync(generateOptions, latents, progressCallback))
                {
                    yield return decodedFrame;
                }
            }
        }


        /// <summary>
        /// Decodes the latents.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="latents">The latents.</param>
        /// <returns></returns>
        protected override async Task<DenseTensor<float>> DecodeLatentsAsync(GenerateOptions options, DenseTensor<float> latents, CancellationToken cancellationToken = default)
        {
            var outputDim = new[] { 1, 3, options.SchedulerOptions.Height, options.SchedulerOptions.Width };
            var metadata = await _vaeDecoder.LoadAsync(cancellationToken: cancellationToken);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
            {
                inferenceParameters.AddInputTensor(latents);
                inferenceParameters.AddOutputBuffer(outputDim);
                var results = await _vaeDecoder.RunInferenceAsync(inferenceParameters);
                using (var imageResult = results.First())
                {
                    if (options.FrameResample)
                        return await ResampleFrameAsync(options, imageResult.ToDenseTensor(), cancellationToken);

                    return imageResult.ToDenseTensor();
                }
            }
        }


        /// <summary>
        /// Resample the frame
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="latents">The latents.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;DenseTensor`1&gt; representing the asynchronous operation.</returns>
        private async Task<DenseTensor<float>> ResampleFrameAsync(GenerateOptions options, DenseTensor<float> latents, CancellationToken cancellationToken = default)
        {
            var downsample = Math.Max(1, options.FrameDownSample);
            var upsampleWidth = options.SchedulerOptions.Width * options.FrameUpSample;
            var upsampleHeight = options.SchedulerOptions.Height * options.FrameUpSample;
            var outputDim = new[] { 1, 3, upsampleHeight, upsampleWidth };
            var metadata = await _resampler.LoadAsync(cancellationToken: cancellationToken);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
            {
                latents.NormalizeOneOneToZeroOne();
                inferenceParameters.AddInputTensor(latents);
                inferenceParameters.AddOutputBuffer(outputDim);
                var results = await _resampler.RunInferenceAsync(inferenceParameters);
                using (var result = results.First())
                {
                    var imageResult = downsample == 1
                        ? result.ToDenseTensor()
                        : result.ToDenseTensor().ResizeImage(upsampleWidth / downsample, upsampleHeight / downsample);
                    imageResult.NormalizeZeroOneToOneOne();
                    return imageResult;
                }
            }
        }


        /// <summary>
        /// Gets the scheduler.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns></returns>
        protected override IScheduler GetScheduler(SchedulerOptions options)
        {
            return options.SchedulerType switch
            {
                SchedulerType.Locomotion => new LocomotionScheduler(options),
                _ => default
            };
        }


        /// <summary>
        /// Decodes the video latents.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="latents">The latents.</param>
        /// <returns></returns>
        protected virtual async IAsyncEnumerable<DenseTensor<float>> DecodeVideoLatentsAsync(GenerateOptions options, DenseTensor<float> latents, IProgress<DiffusionProgress> progressCallback = null)
        {
            var timestamp = _logger.LogBegin();
            var frameCount = latents.Dimensions[2];
            var channels = latents.Dimensions[1];
            var height = latents.Dimensions[3];
            var width = latents.Dimensions[4];
            latents = latents
                .MultiplyBy(1.0f / _vaeDecoder.ScaleFactor)
                .Permute([0, 2, 1, 3, 4])
                .ReshapeTensor([frameCount, channels, height, width]);

            var frameIndex = 0;
            var multiplier = options.Diffuser == DiffuserType.VideoToVideo || options.Diffuser == DiffuserType.ControlNetVideo
                ? (int)(options.InputFrameRate / options.FrameRate)
                : (int)(options.FrameRate / _unet.FrameRate);
            var totalFrames = options.FrameCount * multiplier;
            if (multiplier < 2)
            {
                foreach (var frame in latents.SplitBatch())
                {
                    if (frameIndex >= options.FrameCount)
                        yield break;

                    yield return await DecodeLatentsAsync(options, frame);
                    frameIndex++;

                    ReportProgress(progressCallback, "Generating Frame", frameIndex, options.FrameCount);
                }
            }
            else
            {
                // Intermediate Flow Estimation
                var progressIndex = 0;
                var metadata = await _flowEstimation.LoadAsync();
                var previousDecodedFrame = default(DenseTensor<float>);
                var extraFramePositions = GetFlowEstimationKeyFrames(frameCount, multiplier);
                foreach (var frame in latents.SplitBatch())
                {
                    if (frameIndex >= totalFrames)
                        yield break;

                    var decodedFrame = await DecodeLatentsAsync(options, frame);
                    if (previousDecodedFrame != null)
                    {
                        decodedFrame.NormalizeOneOneToZeroOne();
                        previousDecodedFrame.NormalizeOneOneToZeroOne();
                        var timesteps = extraFramePositions.Contains(frameIndex)
                            ? GetFlowEstimationTimesteps(multiplier)
                            : GetFlowEstimationTimesteps(multiplier - 1);
                        foreach (var timestep in timesteps)
                        {
                            using (var flowEstimationParams = new OnnxInferenceParameters(metadata))
                            {
                                flowEstimationParams.AddInputTensor(previousDecodedFrame);
                                flowEstimationParams.AddInputTensor(decodedFrame);
                                flowEstimationParams.AddInputTensor(CreateTimestepTensor(timestep));
                                flowEstimationParams.AddOutputBuffer(decodedFrame.Dimensions);
                                var results = await _flowEstimation.RunInferenceAsync(flowEstimationParams);
                                using (var result = results.First())
                                {
                                    var middleFrame = result.ToDenseTensor();
                                    middleFrame.NormalizeZeroOneToOneOne();
                                    yield return middleFrame;

                                    progressIndex++;
                                    ReportProgress(progressCallback, "Generating Frame", progressIndex, totalFrames);
                                }
                            }
                        }
                        decodedFrame.NormalizeZeroOneToOneOne();
                    }

                    yield return decodedFrame;
                    previousDecodedFrame = decodedFrame.ToDenseTensor(); // Copy
                    frameIndex++;

                    progressIndex++;
                    ReportProgress(progressCallback, "Generating Frame", progressIndex, totalFrames);
                }

                // Unload if required
                if (options.IsLowMemoryDecoderEnabled)
                    await _flowEstimation.UnloadAsync();
            }

            // Unload if required
            if (options.IsLowMemoryDecoderEnabled)
                await _vaeDecoder.UnloadAsync();

            _logger?.LogEnd(LogLevel.Debug, "VaeDecoder", timestamp);
        }


        /// <summary>
        /// Reports the progress.
        /// </summary>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="message">The message.</param>
        /// <param name="progress">The progress.</param>
        /// <param name="progressMax">The progress maximum.</param>
        /// <param name="elapsed">The elapsed.</param>
        /// <param name="stepLatents">The step latents.</param>
        protected void ReportProgress(IProgress<DiffusionProgress> progressCallback, string message, int progress, int progressMax, long stepTime = 0)
        {
            progressCallback?.Report(new DiffusionProgress
            {
                StepMax = progressMax,
                StepValue = progress,
                Message = $"{message}: {progress:D2}/{progressMax:D2}",
                Elapsed = stepTime > 0 ? Stopwatch.GetElapsedTime(stepTime).TotalMilliseconds : 0.0
            });
        }


        /// <summary>
        /// Performs classifier-free guidance.
        /// </summary>
        /// <param name="noisePredCond">The noise pred cond.</param>
        /// <param name="noisePredUncond">The noise pred uncond.</param>
        /// <param name="guidanceScale">The guidance scale.</param>
        /// <returns></returns>
        protected DenseTensor<float> PerformGuidance(DenseTensor<float> noisePredCond, DenseTensor<float> noisePredUncond, float guidanceScale)
        {
            // Perform guidance
            noisePredUncond.Lerp(noisePredCond, guidanceScale);
            return noisePredUncond;
        }


        /// <summary>
        /// Updates the context window.
        /// </summary>
        /// <param name="noisePrediction">The noise pred.</param>
        /// <param name="prediction">The pred.</param>
        /// <param name="indices">The context.</param>
        protected static void UpdateContextWindow(DenseTensor<float> noisePrediction, DenseTensor<float> prediction, int[] indices)
        {
            var batchSize = noisePrediction.Dimensions[0];
            var channels = noisePrediction.Dimensions[1];
            var height = noisePrediction.Dimensions[3];
            var width = noisePrediction.Dimensions[4];
            Parallel.For(0, indices.Length, ctx =>
            {
                var index = indices[ctx];
                for (int i = 0; i < batchSize; i++)
                {
                    for (int j = 0; j < channels; j++)
                    {
                        for (int h = 0; h < height; h++)
                        {
                            for (int w = 0; w < width; w++)
                            {
                                var value = prediction[i, j, ctx, h, w];
                                var current = noisePrediction[i, j, index, h, w];
                                if (current == 0)
                                {
                                    noisePrediction[i, j, index, h, w] = value;
                                    continue;
                                }

                                // Average the value if it exists
                                noisePrediction[i, j, index, h, w] = (current + value) / 2f;
                            }
                        }
                    }
                }
            });
        }


        /// <summary>
        /// Gets the context window for the specified indices.
        /// </summary>
        /// <param name="latents">The latents.</param>
        /// <param name="indices">The indices.</param>
        /// <returns></returns>
        protected static DenseTensor<float> GetContextWindow(DenseTensor<float> latents, int[] indices)
        {
            var context = indices.Length;
            var batch = latents.Dimensions[0];
            var channels = latents.Dimensions[1];
            var height = latents.Dimensions[3];
            var width = latents.Dimensions[4];
            var result = new DenseTensor<float>([batch, channels, context, height, width]);
            Parallel.For(0, indices.Length, ctx =>
            {
                int index = indices[ctx];
                for (int b = 0; b < batch; b++)
                    for (int c = 0; c < channels; c++)
                        for (int h = 0; h < height; h++)
                            for (int w = 0; w < width; w++)
                                result[b, c, ctx, h, w] = latents[b, c, index, h, w];
            });
            return result;
        }


        /// <summary>
        /// Gets the prompt context window.
        /// </summary>
        /// <param name="promptEmbeds">The prompt embeds.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>DenseTensor&lt;System.Single&gt;.</returns>
        protected static DenseTensor<float> GetPromptContextWindow(DenseTensor<float> promptEmbeds, int[] indices)
        {
            var context = indices.Length;
            var sequence = promptEmbeds.Dimensions[1];
            var size = promptEmbeds.Dimensions[2];
            var result = new DenseTensor<float>([context, sequence, size]);
            Parallel.For(0, indices.Length, ctx =>
            {
                int index = indices[ctx];
                for (int i = 0; i < sequence; i++)
                    for (int h = 0; h < size; h++)
                        result[ctx, i, h] = promptEmbeds[index, i, h];
            });
            return result;
        }


        /// <summary>
        /// Gets the context windows.
        /// </summary>
        /// <param name="step">The step.</param>
        /// <param name="numFrames">The number frames.</param>
        /// <param name="contextSize">Size of the context.</param>
        /// <param name="contextStride">The context stride.</param>
        /// <param name="contextOverlap">The context overlap.</param>
        /// <param name="closedLoop">if set to <c>true</c> [closed loop].</param>
        /// <returns></returns>
        protected static IReadOnlyList<int[]> GetContextWindows(int step, int numFrames, int contextSize = 16, int contextStride = 0, int contextOverlap = 0, bool closedLoop = false)
        {
            if (numFrames <= contextSize)
                return [Enumerable.Range(0, numFrames).ToArray()];

            if (contextStride == 0 && contextOverlap == 0)
                return Enumerable.Range(0, numFrames).Chunk(contextSize).ToList();

            // Adjust contextStride based on log2 calculation
            contextStride = Math.Min(contextStride, (int)MathF.Ceiling(MathF.Log2(numFrames / contextSize)) + 1);

            var results = new List<int[]>();
            for (int contextStep = 1; contextStep <= (1 << contextStride); contextStep <<= 1)
            {
                // Calculate padding and step shifts based on ordered halving
                var orderedHalving = OrderedHalving(step);
                int pad = (int)(numFrames * orderedHalving);
                int start = (int)(orderedHalving * contextStep) + pad;
                int end = numFrames + pad + (closedLoop ? 0 : -contextOverlap);
                int increment = (contextSize * contextStep - contextOverlap);

                for (int j = start; j < end; j += increment)
                {
                    var frameRange = new List<int>();
                    for (int e = j; e < j + contextSize * contextStep; e += contextStep)
                    {
                        frameRange.Add(e % numFrames);
                    }

                    results.Add([.. frameRange]);
                }
            }
            return results;
        }


        /// <summary>
        /// Ordered halving.
        /// </summary>
        /// <param name="val">The value.</param>
        /// <returns></returns>
        private static float OrderedHalving(int val)
        {
            // Convert to unsigned to avoid dealing with sign bits
            // Reverse the bits using bitwise operations
            uint uval = (uint)val;
            uval = ((uval >> 1) & 0x55555555) | ((uval & 0x55555555) << 1);
            uval = ((uval >> 2) & 0x33333333) | ((uval & 0x33333333) << 2);
            uval = ((uval >> 4) & 0x0F0F0F0F) | ((uval & 0x0F0F0F0F) << 4);
            uval = ((uval >> 8) & 0x00FF00FF) | ((uval & 0x00FF00FF) << 8);
            uval = (uval >> 16) | (uval << 16);

            // Return the scaled result
            return MathF.Abs((float)uval / (1UL << 32));
        }


        /// <summary>
        /// Gets the optimizations.
        /// </summary>
        /// <param name="generateOptions">The generate options.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <returns>OnnxOptimizations.</returns>
        protected OnnxOptimizations GetOptimizations(GenerateOptions generateOptions, PromptEmbeddingsResult promptEmbeddings, IProgress<DiffusionProgress> progressCallback = null)
        {
            var optimizations = new OnnxOptimizations(GraphOptimizationLevel.ORT_DISABLE_ALL);
            optimizations.Add("dummy_width", generateOptions.SchedulerOptions.GetScaledWidth());
            optimizations.Add("dummp_height", generateOptions.SchedulerOptions.GetScaledHeight());
            if (_unet.HasOptimizationsChanged(optimizations))
            {
                progressCallback.Notify("Optimizing Pipeline...");
            }
            return optimizations;
        }


        /// <summary>
        /// Gets the frames padded to the next context size.
        /// </summary>
        /// <param name="originalFrames">The original frames.</param>
        /// <param name="contextSize">Size of the context.</param>
        /// <returns>IEnumerable&lt;OnnxImage&gt;.</returns>
        protected IEnumerable<OnnxImage> GetContextFrames(IReadOnlyList<OnnxImage> originalFrames, int contextSize)
        {
            var originalFrameCount = originalFrames.Count;
            var paddedFrameCount = (int)Math.Ceiling(originalFrameCount / (double)contextSize) * contextSize;
            foreach (var frame in originalFrames)
                yield return frame;

            var lastFrame = originalFrames[^1];
            for (int i = originalFrameCount; i < paddedFrameCount; i++)
                yield return lastFrame;
        }


        /// <summary>
        /// Prepares the noise latents.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="frameCount">The frame count.</param>
        /// <returns>Task&lt;DenseTensor&lt;System.Single&gt;&gt;.</returns>
        protected Task<DenseTensor<float>> PrepareNoiseLatentsAsync(GenerateOptions options, IScheduler scheduler, int frameCount)
        {
            var scaledDimension = options.SchedulerOptions.GetScaledDimension();
            var noiseSample = scheduler
                .CreateRandomSample([options.MotionNoiseContext, scaledDimension[1], scaledDimension[2], scaledDimension[3]], scheduler.InitNoiseSigma)
                .Repeat(frameCount / options.MotionNoiseContext)
                .Permute([1, 0, 2, 3])
                .ReshapeTensor([scaledDimension[0], scaledDimension[1], frameCount, scaledDimension[2], scaledDimension[3]]);
            return Task.FromResult(noiseSample);
        }


        /// <summary>
        /// Prepare image latents
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;DenseTensor`1&gt; representing the asynchronous operation.</returns>
        protected async Task<DenseTensor<float>> PrepareImageLatentsAsync(GenerateOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps, CancellationToken cancellationToken = default)
        {
            var imageTensor = await options.InputImage.GetImageTensorAsync(options.SchedulerOptions.Height, options.SchedulerOptions.Width);
            var outputDimension = options.SchedulerOptions.GetScaledDimension();
            var metadata = await _vaeEncoder.LoadAsync(cancellationToken: cancellationToken);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
            {
                inferenceParameters.AddInputTensor(imageTensor);
                inferenceParameters.AddOutputBuffer(outputDimension);

                var results = await _vaeEncoder.RunInferenceAsync(inferenceParameters);
                using (var result = results.First())
                {
                    // Unload if required
                    if (options.IsLowMemoryEncoderEnabled)
                        await _vaeEncoder.UnloadAsync();

                    var outputResult = result.ToDenseTensor();
                    var scaledSample = outputResult
                        .MultiplyBy(_vaeEncoder.ScaleFactor)
                        .Repeat(options.MotionFrames)
                        .Permute([1, 0, 2, 3])
                        .ReshapeTensor([outputResult.Dimensions[0], outputResult.Dimensions[1], options.MotionFrames, outputResult.Dimensions[2], outputResult.Dimensions[3]]);

                    var noise = await PrepareNoiseLatentsAsync(options, scheduler, options.MotionFrames);
                    return scheduler.AddNoise(scaledSample, noise, timesteps);
                }
            }
        }


        /// <summary>
        /// Prepares the video latents.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;DenseTensor`1&gt; representing the asynchronous operation.</returns>
        protected async Task<DenseTensor<float>> PrepareVideoLatentsAsync(GenerateOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps, CancellationToken cancellationToken = default)
        {
            var outputDimension = options.SchedulerOptions.GetScaledDimension();
            var metadata = await _vaeEncoder.LoadAsync(cancellationToken: cancellationToken);

            var scaledLatents = new List<DenseTensor<float>>();
            foreach (var frame in GetContextFrames(options.InputVideo.Frames, _unet.ContextSize))
            {
                var imageTensor = await frame.GetImageTensorAsync(options.SchedulerOptions.Height, options.SchedulerOptions.Width);
                using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
                {
                    inferenceParameters.AddInputTensor(imageTensor);
                    inferenceParameters.AddOutputBuffer(outputDimension);

                    var results = await _vaeEncoder.RunInferenceAsync(inferenceParameters);
                    using (var result = results.First())
                    {
                        var outputResult = result.ToDenseTensor();
                        var scaledSample = outputResult.MultiplyBy(_vaeEncoder.ScaleFactor);
                        scaledLatents.Add(scaledSample);
                    }
                }
            }

            // Unload if required
            if (options.IsLowMemoryEncoderEnabled)
                await _vaeEncoder.UnloadAsync();

            var noiseLatents = new List<DenseTensor<float>>();
            var noise = scheduler.CreateRandomSample([options.MotionNoiseContext, outputDimension[1], outputDimension[2], outputDimension[3]]);
            foreach (var contextChunk in scaledLatents.Chunk(options.MotionNoiseContext))
            {
                noiseLatents.Add(scheduler.AddNoise(contextChunk.Join(), noise, timesteps));
            }

            var latents = noiseLatents.Join();
            return noiseLatents.Join()
                .Permute([1, 0, 2, 3])
                .ReshapeTensor([1, latents.Dimensions[1], latents.Dimensions[0], latents.Dimensions[2], latents.Dimensions[3]]);
        }


        /// <summary>
        /// Creates the timestep tensor.
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <returns>DenseTensor&lt;System.Single&gt;.</returns>
        private static DenseTensor<float> CreateTimestepTensor(float timestep)
        {
            return new DenseTensor<float>(new float[] { timestep }, new int[] { 1 });
        }


        /// <summary>
        /// Gets the flow estimation timesteps.
        /// </summary>
        /// <param name="parts">The parts.</param>
        /// <returns>System.Single[].</returns>
        private static float[] GetFlowEstimationTimesteps(int parts)
        {
            float[] result = new float[parts];

            for (int i = 0; i < parts; i++)
            {
                result[i] = (i + 1) / (float)(parts + 1);
            }

            return result;
        }


        /// <summary>
        /// Gets the flow estimation key frames.
        /// </summary>
        /// <param name="frameCount">The frame count.</param>
        /// <param name="multiplier">The multiplier.</param>
        /// <returns>System.Int32[].</returns>
        private static int[] GetFlowEstimationKeyFrames(int frameCount, int multiplier)
        {
            int targetCount = frameCount * multiplier;
            int extraFramesNeeded = targetCount - (frameCount - 1) * (multiplier - 1) - frameCount;

            if (multiplier == 2)
                return new[] { frameCount - 1 };

            else if (multiplier == 3)
                return new[] { 0, frameCount / 2, frameCount - 1 };

            return Enumerable.Range(0, multiplier)
                .Select(i => (int)Math.Round(i * (frameCount - 1) / (double)(multiplier - 1)))
                .Distinct()
                .ToArray();
        }
    }
}
