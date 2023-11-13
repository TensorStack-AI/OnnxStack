using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
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
        protected abstract Task<DenseTensor<float>> PrepareLatents(IModelOptions model, PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps);


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
        protected abstract Task<DenseTensor<float>> SchedulerStep(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, DenseTensor<float> promptEmbeddings, bool performGuidance, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default);


        /// <summary>
        /// Rund the stable diffusion loop
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="progress">The progress.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public virtual async Task<DenseTensor<float>> DiffuseAsync(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default)
        {
            // Create random seed if none was set
            schedulerOptions.Seed = schedulerOptions.Seed > 0 ? schedulerOptions.Seed : Random.Shared.Next();

            var diffuseTime = _logger?.LogBegin("Begin...");
            _logger?.Log($"Model: {modelOptions.Name}, Pipeline: {modelOptions.PipelineType}, Diffuser: {promptOptions.DiffuserType}, Scheduler: {schedulerOptions.SchedulerType}");

            // Check guidance
            var performGuidance = ShouldPerformGuidance(schedulerOptions);

            // Process prompts
            var promptEmbeddings = await _promptService.CreatePromptAsync(modelOptions, promptOptions, performGuidance);

            // Run Scheduler steps
            var schedulerResult = await SchedulerStep(modelOptions, promptOptions, schedulerOptions, promptEmbeddings, performGuidance, progressCallback, cancellationToken);

            _logger?.LogEnd($"End", diffuseTime);

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
        public virtual async IAsyncEnumerable<BatchResult> DiffuseBatchAsync(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<int, int, int, int> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            // Create random seed if none was set
            schedulerOptions.Seed = schedulerOptions.Seed > 0 ? schedulerOptions.Seed : Random.Shared.Next();

            var diffuseBatchTime = _logger?.LogBegin("Begin...");
            _logger?.Log($"Model: {modelOptions.Name}, Pipeline: {modelOptions.PipelineType}, Diffuser: {promptOptions.DiffuserType}, Scheduler: {schedulerOptions.SchedulerType}");

            // Check guidance
            var performGuidance = ShouldPerformGuidance(schedulerOptions);

            // Process prompts
            var promptEmbeddings = await _promptService.CreatePromptAsync(modelOptions, promptOptions, performGuidance);

            // Generate batch options
            var batchSchedulerOptions = BatchGenerator.GenerateBatch(modelOptions, batchOptions, schedulerOptions);

            var batchIndex = 1;
            var schedulerCallback = (int step, int steps) => progressCallback?.Invoke(batchIndex, batchSchedulerOptions.Count, step, steps);
            foreach (var batchSchedulerOption in batchSchedulerOptions)
            {
                yield return new BatchResult(batchSchedulerOption, await SchedulerStep(modelOptions, promptOptions, batchSchedulerOption, promptEmbeddings, performGuidance, schedulerCallback, cancellationToken));
                batchIndex++;
            }

            _logger?.LogEnd($"End", diffuseBatchTime);
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
        protected virtual async Task<DenseTensor<float>> DecodeLatents(IModelOptions model, PromptOptions prompt, SchedulerOptions options, DenseTensor<float> latents)
        {
            var timestamp = _logger?.LogBegin("Begin...");

            // Scale and decode the image latents with vae.
            latents = latents.MultiplyBy(1.0f / model.ScaleFactor);

            var images = prompt.BatchCount > 1
                ? latents.Split(prompt.BatchCount)
                : new[] { latents };
            var imageTensors = new List<DenseTensor<float>>();
            foreach (var image in images)
            {
                var inputNames = _onnxModelService.GetInputNames(model, OnnxModelType.VaeDecoder);
                var inputParameters = CreateInputParameters(NamedOnnxValue.CreateFromTensor(inputNames[0], image));

                // Run inference.
                using (var inferResult = await _onnxModelService.RunInferenceAsync(model, OnnxModelType.VaeDecoder, inputParameters))
                {
                    var resultTensor = inferResult.FirstElementAs<DenseTensor<float>>();
                    imageTensors.Add(resultTensor.ToDenseTensor());
                }
            }

            var result = prompt.BatchCount > 1
                ? imageTensors.Join()
                : imageTensors.FirstOrDefault();
            _logger?.LogEnd("End", timestamp);
            return result;
        }


        /// <summary>
        /// Creates the timestep NamedOnnxValue based on its NodeMetadata type.
        /// </summary>
        /// <param name="nodeMetadata">The node metadata.</param>
        /// <param name="timestepInputName">Name of the timestep input.</param>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        protected static NamedOnnxValue CreateTimestepNamedOnnxValue(IReadOnlyDictionary<string, NodeMetadata> nodeMetadata, string timestepInputName, int timestep)
        {
            // Some models support Long or Float, could be more but fornow just support these 2
            var timestepMetaData = nodeMetadata[timestepInputName];
            return timestepMetaData.ElementDataType == TensorElementType.Int64
                ? NamedOnnxValue.CreateFromTensor(timestepInputName, new DenseTensor<long>(new long[] { timestep }, new int[] { 1 }))
                : NamedOnnxValue.CreateFromTensor(timestepInputName, new DenseTensor<float>(new float[] { timestep }, new int[] { 1 }));
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
    }
}
