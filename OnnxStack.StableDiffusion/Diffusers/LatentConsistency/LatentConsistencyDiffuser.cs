﻿using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Schedulers.LatentConsistency;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers.LatentConsistency
{
    public abstract class LatentConsistencyDiffuser : IDiffuser
    {
        protected readonly IPromptService _promptService;
        protected readonly IOnnxModelService _onnxModelService;
        protected readonly ILogger<LatentConsistencyDiffuser> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="LatentConsistencyDiffuser"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="onnxModelService">The onnx model service.</param>
        public LatentConsistencyDiffuser(IOnnxModelService onnxModelService, IPromptService promptService, ILogger<LatentConsistencyDiffuser> logger)
        {
            _logger = logger;
            _promptService = promptService;
            _onnxModelService = onnxModelService;
        }


        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public DiffuserPipelineType PipelineType => DiffuserPipelineType.LatentConsistency;


        /// <summary>
        /// Gets the type of the diffuser.
        /// </summary>
        public abstract DiffuserType DiffuserType { get; }


        /// <summary>
        /// Gets the timesteps.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <returns></returns>
        protected abstract IReadOnlyList<int> GetTimesteps(PromptOptions prompt, SchedulerOptions options, IScheduler scheduler);

        /// <summary>
        /// Prepares the latents.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <returns></returns>
        protected abstract DenseTensor<float> PrepareLatents(IModelOptions model, PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps);


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
            _logger?.Log($"Model: {modelOptions.Name}, Pipeline: {modelOptions.PipelineType}, Diffuser: {promptOptions.DiffuserType}, Scheduler: {promptOptions.SchedulerType}");

            // LCM does not support negative prompting
            promptOptions.NegativePrompt = string.Empty;

            // Get Scheduler
            using (var scheduler = GetScheduler(promptOptions, schedulerOptions))
            {
                // Process prompts
                var promptEmbeddings = await _promptService.CreatePromptAsync(modelOptions, promptOptions, false);

                // Get timesteps
                var timesteps = GetTimesteps(promptOptions, schedulerOptions, scheduler);

                // Create latent sample
                var latents = PrepareLatents(modelOptions, promptOptions, schedulerOptions, scheduler, timesteps);

                // Get Guidance Scale Embedding
                var guidanceEmbeddings = GetGuidanceScaleEmbedding(schedulerOptions.GuidanceScale);

                // Denoised result
                DenseTensor<float> denoised = null;

                var unetSession = _onnxModelService.GetOnnxSession(OnnxModelType.Unet);

                // Loop though the timesteps
                var step = 0;
                foreach (var timestep in timesteps)
                {
                    step++;
                    var stepTime = Stopwatch.GetTimestamp();
                    cancellationToken.ThrowIfCancellationRequested();

                    // Create input tensor.
                    var inputTensor = scheduler.ScaleInput(latents, timestep);

                    // Create Input Parameters
                    var inputParameters = CreateUnetInputParams(modelOptions, inputTensor, promptEmbeddings, guidanceEmbeddings, timestep, unetSession);

                    // Run Inference
                    using (var inferResult = await _onnxModelService.RunInferenceAsync(modelOptions, OnnxModelType.Unet, inputParameters))
                    {
                        var noisePred = inferResult.FirstElementAs<DenseTensor<float>>();

                        // Scheduler Step
                        var schedulerResult = scheduler.Step(noisePred, timestep, latents);

                        latents = schedulerResult.Result;
                        denoised = schedulerResult.SampleData;
                    }

                    progressCallback?.Invoke(step, timesteps.Count);
                    _logger?.LogEnd(LogLevel.Debug, $"Step {step}/{timesteps.Count}", stepTime);
                }

                // Decode Latents
                var result = await DecodeLatents(modelOptions, promptOptions, schedulerOptions, denoised);
                _logger?.LogEnd($"End", diffuseTime);
                return result;
            }
        }


        /// <summary>
        /// Decodes the latents.
        /// </summary>
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
        /// Creates the Unet input parameters.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="inputTensor">The input tensor.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        protected virtual IReadOnlyList<NamedOnnxValue> CreateUnetInputParams(IModelOptions model, DenseTensor<float> inputTensor, DenseTensor<float> promptEmbeddings, DenseTensor<float> guidanceEmbeddings, int timestep, InferenceSession unetSession)
        {
            var inputNames = _onnxModelService.GetInputNames(model, OnnxModelType.Unet);
            return CreateInputParameters(
                 NamedOnnxValue.CreateFromTensor(inputNames[0], inputTensor),
                 GetTimestep(model, timestep, inputNames, unetSession),
                 NamedOnnxValue.CreateFromTensor(inputNames[2], promptEmbeddings),
                 NamedOnnxValue.CreateFromTensor(inputNames[3], guidanceEmbeddings));
        }

        protected virtual NamedOnnxValue GetTimestep(IModelOptions model, int timestep, IReadOnlyList<string> inputNames, InferenceSession unetSession)
        {
            TensorElementType elementDataType = unetSession.InputMetadata["timestep"].ElementDataType;
            if (elementDataType == TensorElementType.Float)
                return NamedOnnxValue.CreateFromTensor(inputNames[1], new DenseTensor<float>(new float[] { timestep }, new int[] { 1 }));
            if (elementDataType == TensorElementType.Int64)
                return NamedOnnxValue.CreateFromTensor(inputNames[1], new DenseTensor<long>(new long[] { timestep }, new int[] { 1 }));
            throw new Exception($"Unable to map timestep to tensor value `{elementDataType}`");
        }

        /// <summary>
        /// Gets the scheduler.
        /// </summary>
        /// <param name="prompt"></param>
        /// <param name="options">The options.</param>
        /// <returns></returns>
        protected IScheduler GetScheduler(PromptOptions prompt, SchedulerOptions options)
        {
            return prompt.SchedulerType switch
            {
                SchedulerType.LCM => new LCMScheduler(options),
                _ => default
            };
        }


        /// <summary>
        /// Gets the guidance scale embedding.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="embeddingDim">The embedding dim.</param>
        /// <returns></returns>
        private DenseTensor<float> GetGuidanceScaleEmbedding(float guidance, int embeddingDim = 256)
        {
            var scale = guidance - 1f;
            var halfDim = embeddingDim / 2;
            float log = MathF.Log(10000.0f) / (halfDim - 1);
            var emb = Enumerable.Range(0, halfDim)
                .Select(x => MathF.Exp(x * -log))
                .ToArray();
            var embSin = emb.Select(MathF.Sin).ToArray();
            var embCos = emb.Select(MathF.Cos).ToArray();
            var result = new DenseTensor<float>(new[] { 1, 2 * halfDim });
            for (int i = 0; i < halfDim; i++)
            {
                result[0, i] = embSin[i];
                result[0, i + halfDim] = embCos[i];
            }
            return result;
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
