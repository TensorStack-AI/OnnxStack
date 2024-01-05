using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers
{
    public abstract class DiffuserBase : IDiffuser
    {
        protected readonly IPromptService _promptService;
        protected readonly IOnnxModelService _onnxModelService;
        protected readonly ILogger<DiffuserBase> _logger;


        /// <summary>
        /// Initializes a new instance of the <see cref="DiffuserBase"/> class.
        /// </summary>
        /// <param name="onnxModelService">The onnx model service.</param>
        /// <param name="promptService">The prompt service.</param>
        /// <param name="logger">The logger.</param>
        public DiffuserBase(IOnnxModelService onnxModelService, IPromptService promptService, ILogger<DiffuserBase> logger)
        {
            _logger = logger;
            _promptService = promptService;
            _onnxModelService = onnxModelService;
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
        /// <param name="model">The model.</param>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <returns></returns>
        protected abstract Task<DenseTensor<float>> PrepareLatentsAsync(ModelOptions model, PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps);


        /// <summary>
        /// Called on each Scheduler step.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <param name="performGuidance">if set to <c>true</c> [perform guidance].</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        protected abstract Task<DenseTensor<float>> SchedulerStepAsync(ModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, PromptEmbeddingsResult promptEmbeddings, bool performGuidance, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Rund the stable diffusion loop
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="progress">The progress.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public virtual async Task<DenseTensor<float>> DiffuseAsync(ModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            // Create random seed if none was set
            schedulerOptions.Seed = schedulerOptions.Seed > 0 ? schedulerOptions.Seed : Random.Shared.Next();

            var diffuseTime = _logger?.LogBegin("Diffuse starting...");
            _logger?.Log($"Model: {modelOptions.Name}, Pipeline: {modelOptions.PipelineType}, Diffuser: {promptOptions.DiffuserType}, Scheduler: {schedulerOptions.SchedulerType}");

            // Check guidance
            var performGuidance = ShouldPerformGuidance(schedulerOptions);

            // Process prompts
            var promptEmbeddings = await _promptService.CreatePromptAsync(modelOptions.BaseModel, promptOptions, performGuidance);

            // If video input, process frames
            if (promptOptions.HasInputVideo)
            {
                var frameIndex = 0;
                DenseTensor<float> videoTensor = null;
                var videoFrames = promptOptions.InputVideo.VideoFrames.Frames;
                var schedulerFrameCallback = CreateBatchCallback(progressCallback, videoFrames.Count, () => frameIndex);
                foreach (var videoFrame in videoFrames)
                {
                    frameIndex++;
                    promptOptions.InputImage = promptOptions.DiffuserType == DiffuserType.ControlNet ? default : new InputImage(videoFrame);
                    promptOptions.InputContolImage = promptOptions.DiffuserType == DiffuserType.ImageToImage ? default : new InputImage(videoFrame);
                    var frameResultTensor = await SchedulerStepAsync(modelOptions, promptOptions, schedulerOptions, promptEmbeddings, performGuidance, schedulerFrameCallback, cancellationToken);

                    // Frame Progress
                    ReportBatchProgress(progressCallback, frameIndex, videoFrames.Count, frameResultTensor);

                    // Concatenate frame
                    videoTensor = videoTensor.Concatenate(frameResultTensor);
                }

                _logger?.LogEnd($"Diffuse complete", diffuseTime);
                return videoTensor;
            }

            // Run Scheduler steps
            var schedulerResult = await SchedulerStepAsync(modelOptions, promptOptions, schedulerOptions, promptEmbeddings, performGuidance, progressCallback, cancellationToken);
            _logger?.LogEnd($"Diffuse complete", diffuseTime);
            return schedulerResult;
        }



        /// <summary>
        /// Runs the stable diffusion batch loop
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="batchOptions">The batch options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        /// <exception cref="System.NotImplementedException"></exception>
        public virtual async IAsyncEnumerable<BatchResult> DiffuseBatchAsync(ModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<DiffusionProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            // Create random seed if none was set
            schedulerOptions.Seed = schedulerOptions.Seed > 0 ? schedulerOptions.Seed : Random.Shared.Next();

            var diffuseBatchTime = _logger?.LogBegin("Batch Diffuser starting...");
            _logger?.Log($"Model: {modelOptions.Name}, Pipeline: {modelOptions.PipelineType}, Diffuser: {promptOptions.DiffuserType}, Scheduler: {schedulerOptions.SchedulerType}");
            _logger?.Log($"BatchType: {batchOptions.BatchType}, ValueFrom: {batchOptions.ValueFrom}, ValueTo: {batchOptions.ValueTo}, Increment: {batchOptions.Increment}");

            // Check guidance
            var performGuidance = ShouldPerformGuidance(schedulerOptions);

            // Process prompts
            var promptEmbeddings = await _promptService.CreatePromptAsync(modelOptions.BaseModel, promptOptions, performGuidance);

            // Generate batch options
            var batchSchedulerOptions = BatchGenerator.GenerateBatch(modelOptions, batchOptions, schedulerOptions);

            var batchIndex = 1;
            var batchSchedulerCallback = CreateBatchCallback(progressCallback, batchSchedulerOptions.Count, () => batchIndex);
            foreach (var batchSchedulerOption in batchSchedulerOptions)
            {
                var diffuseTime = _logger?.LogBegin("Diffuse starting...");
                yield return new BatchResult(batchSchedulerOption, await SchedulerStepAsync(modelOptions, promptOptions, batchSchedulerOption, promptEmbeddings, performGuidance, batchSchedulerCallback, cancellationToken));
                _logger?.LogEnd($"Diffuse complete", diffuseTime);
                batchIndex++;
            }

            _logger?.LogEnd($"Diffuse batch complete", diffuseBatchTime);
        }


        /// <summary>
        /// Chech if we should run guidance.
        /// </summary>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <returns></returns>
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
        /// <param name="model">The model.</param>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="latents">The latents.</param>
        /// <returns></returns>
        protected virtual async Task<DenseTensor<float>> DecodeLatentsAsync(ModelOptions model, PromptOptions prompt, SchedulerOptions options, DenseTensor<float> latents)
        {
            var timestamp = _logger.LogBegin();

            // Scale and decode the image latents with vae.
            latents = latents.MultiplyBy(1.0f / model.ScaleFactor);

            var outputDim = new[] { 1, 3, options.Height, options.Width };
            var metadata = _onnxModelService.GetModelMetadata(model.BaseModel, OnnxModelType.VaeDecoder);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(latents);
                inferenceParameters.AddOutputBuffer(outputDim);

                var results = await _onnxModelService.RunInferenceAsync(model.BaseModel, OnnxModelType.VaeDecoder, inferenceParameters);
                using (var imageResult = results.First())
                {
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
        /// Helper for creating the input parameters.
        /// </summary>
        /// <param name="parameters">The parameters.</param>
        /// <returns></returns>
        protected static IReadOnlyList<NamedOnnxValue> CreateInputParameters(params NamedOnnxValue[] parameters)
        {
            return parameters.ToList();
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


        /// <summary>
        /// Reports the progress.
        /// </summary>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="progress">The progress.</param>
        /// <param name="progressMax">The progress maximum.</param>
        /// <param name="subProgress">The sub progress.</param>
        /// <param name="subProgressMax">The sub progress maximum.</param>
        /// <param name="output">The output.</param>
        protected void ReportBatchProgress(Action<DiffusionProgress> progressCallback, int progress, int progressMax, DenseTensor<float> progressTensor)
        {
            progressCallback?.Invoke(new DiffusionProgress
            {
                BatchMax = progressMax,
                BatchValue = progress,
                BatchTensor = progressTensor
            });
        }


        private static Action<DiffusionProgress> CreateBatchCallback(Action<DiffusionProgress> progressCallback, int batchCount, Func<int> batchIndex)
        {
            if (progressCallback == null)
                return progressCallback;

            return (DiffusionProgress progress) => progressCallback?.Invoke(new DiffusionProgress
            {
                StepMax = progress.StepMax,
                StepValue = progress.StepValue,
                StepTensor = progress.StepTensor,
                BatchMax = batchCount,
                BatchValue = batchIndex()
            });
        }

    }
}
