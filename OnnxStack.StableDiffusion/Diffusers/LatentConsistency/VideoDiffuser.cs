using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers.LatentConsistency
{
    public sealed class VideoDiffuser : LatentConsistencyDiffuser
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="VideoDiffuser"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="onnxModelService">The onnx model service.</param>
        public VideoDiffuser(IOnnxModelService onnxModelService, IPromptService promptService, ILogger<LatentConsistencyDiffuser> logger)
            : base(onnxModelService, promptService, logger) { }


        /// <summary>
        /// Gets the type of the diffuser.
        /// </summary>
        public override DiffuserType DiffuserType => DiffuserType.VideoToVideo;


        /// <summary>
        /// Runs the scheduler steps.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <param name="performGuidance">if set to <c>true</c> [perform guidance].</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        protected override async Task<DenseTensor<float>> SchedulerStepAsync(StableDiffusionModelSet modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, PromptEmbeddingsResult promptEmbeddings, bool performGuidance, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default)
        {
            DenseTensor<float> resultTensor = null;
            foreach (var videoFrame in promptOptions.InputVideo.Frames)
            {
                // Get Scheduler
                using (var scheduler = GetScheduler(schedulerOptions))
                {
                    // Get timesteps
                    var timesteps = GetTimesteps(schedulerOptions, scheduler);

                    // Create latent sample
                    var latents = await PrepareFrameLatentsAsync(modelOptions, videoFrame, schedulerOptions, scheduler, timesteps);

                    // Get Guidance Scale Embedding
                    var guidanceEmbeddings = GetGuidanceScaleEmbedding(schedulerOptions.GuidanceScale);

                    // Denoised result
                    DenseTensor<float> denoised = null;

                    // Get Model metadata
                    var metadata = _onnxModelService.GetModelMetadata(modelOptions, OnnxModelType.Unet);

                    // Loop though the timesteps
                    var step = 0;
                    foreach (var timestep in timesteps)
                    {
                        step++;
                        var stepTime = Stopwatch.GetTimestamp();
                        cancellationToken.ThrowIfCancellationRequested();

                        // Create input tensor.
                        var inputTensor = scheduler.ScaleInput(latents, timestep);
                        var timestepTensor = CreateTimestepTensor(timestep);

                        var outputChannels = 1;
                        var outputDimension = schedulerOptions.GetScaledDimension(outputChannels);
                        using (var inferenceParameters = new OnnxInferenceParameters(metadata))
                        {
                            inferenceParameters.AddInputTensor(inputTensor);
                            inferenceParameters.AddInputTensor(timestepTensor);
                            inferenceParameters.AddInputTensor(promptEmbeddings.PromptEmbeds);
                            inferenceParameters.AddInputTensor(guidanceEmbeddings);
                            inferenceParameters.AddOutputBuffer(outputDimension);

                            var results = await _onnxModelService.RunInferenceAsync(modelOptions, OnnxModelType.Unet, inferenceParameters);
                            using (var result = results.First())
                            {
                                var noisePred = result.ToDenseTensor();

                                // Scheduler Step
                                var schedulerResult = scheduler.Step(noisePred, timestep, latents);

                                latents = schedulerResult.Result;
                                denoised = schedulerResult.SampleData;
                            }
                        }

                        progressCallback?.Invoke(step, timesteps.Count);
                        _logger?.LogEnd($"Step {step}/{timesteps.Count}", stepTime);
                    }

                    // Decode Latents
                    var frameResultTensor = await DecodeLatentsAsync(modelOptions, promptOptions, schedulerOptions, denoised);
                    resultTensor = resultTensor is null
                        ? frameResultTensor
                        : resultTensor.Concatenate(frameResultTensor);
                }
            }
            return resultTensor;
        }


        /// <summary>
        /// Gets the timesteps.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <returns></returns>
        protected override IReadOnlyList<int> GetTimesteps(SchedulerOptions options, IScheduler scheduler)
        {
            var inittimestep = Math.Min((int)(options.InferenceSteps * options.Strength), options.InferenceSteps);
            var start = Math.Max(options.InferenceSteps - inittimestep, 0);
            return scheduler.Timesteps.Skip(start).ToList();
        }


        /// <summary>
        /// Prepares the latents for inference.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <returns></returns>
        private async Task<DenseTensor<float>> PrepareFrameLatentsAsync(StableDiffusionModelSet model, byte[] videoFrame, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps)
        {
            var imageTensor = ImageHelpers.TensorFromBytes(videoFrame, new[] { 1, 3, options.Height, options.Width });

            //TODO: Model Config, Channels
            var outputDimension = options.GetScaledDimension();
            var metadata = _onnxModelService.GetModelMetadata(model, OnnxModelType.VaeEncoder);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(imageTensor);
                inferenceParameters.AddOutputBuffer(outputDimension);

                var results = await _onnxModelService.RunInferenceAsync(model, OnnxModelType.VaeEncoder, inferenceParameters);
                using (var result = results.First())
                {
                    var outputResult = result.ToDenseTensor();
                    var scaledSample = outputResult.MultiplyBy(model.ScaleFactor);
                    return scheduler.AddNoise(scaledSample, scheduler.CreateRandomSample(scaledSample.Dimensions), timesteps);
                }
            }
        }

        protected override Task<DenseTensor<float>> PrepareLatentsAsync(StableDiffusionModelSet model, PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps)
        {
            throw new NotImplementedException();
        }
    }
}
