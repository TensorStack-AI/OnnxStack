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
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers.Locomotion
{
    public class ControlNetDiffuser : TextDiffuser
    {
        protected ControlNetModel _controlNet;

        /// <summary>
        /// Initializes a new instance of the <see cref="ControlNetDiffuser"/> class.
        /// </summary>
        /// <param name="controlNet">The unet.</param>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="memoryMode"></param>
        /// <param name="logger">The logger.</param>
        public ControlNetDiffuser(ControlNetModel controlNet, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, FlowEstimationModel flowEstimation, ResampleModel resampler, ILogger logger = default)
            : base(unet, vaeDecoder, vaeEncoder, flowEstimation, resampler, logger)
        {
            _controlNet = controlNet;
        }


        /// <summary>
        /// Creates the Conditioning Scale tensor.
        /// </summary>
        /// <param name="conditioningScale">The conditioningScale.</param>
        /// <returns></returns>
        protected static DenseTensor<double> CreateConditioningScaleTensor(float conditioningScale)
        {
            return new DenseTensor<double>(new double[] { conditioningScale }, new int[] { 1 });
        }


        /// <summary>
        /// Prepares the control image.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns></returns>
        protected virtual async Task<DenseTensor<float>> PrepareControlLatents(GenerateOptions options)
        {
            var controlImageTensor = await options.InputContolImage.GetImageTensorAsync(options.SchedulerOptions.Height, options.SchedulerOptions.Width, ImageNormalizeType.ZeroToOne);
            if (_controlNet.InvertInput)
                InvertInputTensor(controlImageTensor);

            return controlImageTensor.Repeat(options.MotionFrames);
        }


        /// <summary>
        /// Runs the Locomotion Diffusion process
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>A Task&lt;IAsyncEnumerable`1&gt; representing the asynchronous operation.</returns>
        protected override async IAsyncEnumerable<DenseTensor<float>> DiffuseInternalAsync(DiffuseOptions options, IProgress<DiffusionProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
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
                var controlNetMetadata = await _controlNet.LoadAsync(cancellationToken: cancellationToken);

                // Create latent sample
                progressCallback.Notify("Prepare Input Frames...");
                var latents = await PrepareLatentsAsync(generateOptions, scheduler, timesteps, cancellationToken);
                var controlLatents = await PrepareControlLatents(generateOptions);

                // Setup motion/context parameters
                var totalFrames = latents.Dimensions[2];
                var contextCount = totalFrames / contextSize;
                var contextStride = generateOptions.MotionStrides;
                var contextOverlap = generateOptions.MotionContextOverlap;
                var contextWindowCount = GetContextWindows(0, totalFrames, contextSize, contextStride, contextOverlap).Count;

                var step = 0;
                var contextStep = 0;
                var contextSteps = timesteps.Count * contextWindowCount;
                ReportProgress(progressCallback, "Step", 0, contextSteps);
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
                        var scaledLatents = scheduler.ScaleInput(latents, timestep);
                        var contextLatents = GetContextWindow(scaledLatents, contextWindow);
                        var contextControlLatents = GetControlNetContextWindow(controlLatents, contextWindow);
                        var contextControlTensor = GetControlNetInput(contextLatents);
                        var contextPromptCond = GetPromptContextWindow(promptEmbeddingsCond, contextWindow);
                        var contextPromptUncond = GetPromptContextWindow(promptEmbeddingsUncond, contextWindow);
                        var conditioningScale = CreateConditioningScaleTensor(schedulerOptions.ConditioningScale);

                        using (var unetCondParams = new OnnxInferenceParameters(metadata, cancellationToken))
                        using (var unetUncondParams = new OnnxInferenceParameters(metadata, cancellationToken))
                        using (var cnCondParams = new OnnxInferenceParameters(controlNetMetadata, cancellationToken))
                        using (var cnUncondParams = new OnnxInferenceParameters(controlNetMetadata, cancellationToken))
                        {
                            // Conditional Pass
                            unetCondParams.AddInputTensor(contextLatents);
                            unetCondParams.AddInputTensor(timestepTensor);
                            unetCondParams.AddInputTensor(contextPromptCond);

                            // ControlNet
                            cnCondParams.AddInputTensor(contextControlTensor);
                            cnCondParams.AddInputTensor(timestepTensor);
                            cnCondParams.AddInputTensor(contextPromptCond);
                            cnCondParams.AddInputTensor(contextControlLatents);
                            if (controlNetMetadata.Inputs.Count == 5)
                                cnCondParams.AddInputTensor(conditioningScale);

                            // Optimization: Pre-allocate device buffers for inputs
                            foreach (var item in controlNetMetadata.Outputs)
                                cnCondParams.AddOutputBuffer();

                            // ControlNet inference
                            var cnCondResults = _controlNet.RunInference(cnCondParams);

                            // Add ControlNet outputs to Unet input
                            foreach (var item in cnCondResults)
                                unetCondParams.AddInput(item);

                            unetCondParams.AddOutputBuffer(contextLatents.Dimensions);
                            var unetCondResults = await _unet.RunInferenceAsync(unetCondParams);


                            // Unconditional Pass
                            var unetUncondResults = default(IReadOnlyCollection<OrtValue>);
                            if (performGuidance)
                            {
                                unetUncondParams.AddInputTensor(contextLatents);
                                unetUncondParams.AddInputTensor(timestepTensor);
                                unetUncondParams.AddInputTensor(contextPromptUncond);

                                // ControlNet
                                cnUncondParams.AddInputTensor(contextControlTensor);
                                cnUncondParams.AddInputTensor(timestepTensor);
                                cnUncondParams.AddInputTensor(contextPromptUncond);
                                cnUncondParams.AddInputTensor(contextControlLatents);
                                if (controlNetMetadata.Inputs.Count == 5)
                                    cnUncondParams.AddInputTensor(conditioningScale);

                                // Optimization: Pre-allocate device buffers for inputs
                                foreach (var item in controlNetMetadata.Outputs)
                                    cnUncondParams.AddOutputBuffer();

                                // ControlNet inference
                                var cnUncondResults = _controlNet.RunInference(cnUncondParams);

                                // Add ControlNet outputs to Unet input
                                foreach (var item in cnUncondResults)
                                    unetUncondParams.AddInput(item);

                                unetUncondParams.AddOutputBuffer(contextLatents.Dimensions);
                                unetUncondResults = await _unet.RunInferenceAsync(unetUncondParams);
                            }


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
                    await Task.WhenAll(_controlNet.UnloadAsync(), _unet.UnloadAsync());

                // Decode Latents
                await foreach (var decodedFrame in DecodeVideoLatentsAsync(generateOptions, latents))
                {
                    yield return decodedFrame;
                }
            }
        }


        /// <summary>
        /// Gets the control net input.
        /// </summary>
        /// <param name="inputTensor">The input tensor.</param>
        /// <returns>DenseTensor&lt;System.Single&gt;.</returns>
        private DenseTensor<float> GetControlNetInput(DenseTensor<float> inputTensor)
        {
            return inputTensor
                 .Permute([0, 2, 1, 3, 4])
                 .ReshapeTensor([inputTensor.Dimensions[2], inputTensor.Dimensions[1], inputTensor.Dimensions[3], inputTensor.Dimensions[4]]);
        }


        /// <summary>
        /// Gets the control net context window.
        /// </summary>
        /// <param name="latents">The latents.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>DenseTensor&lt;System.Single&gt;.</returns>
        protected static DenseTensor<float> GetControlNetContextWindow(DenseTensor<float> latents, int[] indices)
        {
            var context = indices.Length;
            var batch = latents.Dimensions[0];
            var channels = latents.Dimensions[1];
            var height = latents.Dimensions[2];
            var width = latents.Dimensions[3];
            var result = new DenseTensor<float>([context, channels, height, width]);
            Parallel.For(0, context, ctx =>
            {
                int index = indices[ctx];
                for (int c = 0; c < channels; c++)
                    for (int h = 0; h < height; h++)
                        for (int w = 0; w < width; w++)
                            result[ctx, c, h, w] = latents[index, c, h, w];
            });
            return result;
        }


        /// <summary>
        /// Inverts the input tensor.
        /// </summary>
        /// <param name="values">The values.</param>
        protected static void InvertInputTensor(DenseTensor<float> values)
        {
            for (int j = 0; j < values.Length; j++)
            {
                values.SetValue(j, 1f - values.GetValue(j));
            }
        }
    }
}
