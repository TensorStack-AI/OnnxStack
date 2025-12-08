using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers.StableDiffusionXL;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.StableDiffusion.Schedulers.StableDiffusion;
using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers.Flux
{
    public abstract class FluxDiffuser : StableDiffusionXLDiffuser
    {

        /// <summary>
        /// Initializes a new instance of the <see cref="FluxDiffuser"/> class.
        /// </summary>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        protected FluxDiffuser(UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, ILogger logger = default)
            : base(unet, vaeDecoder, vaeEncoder, logger) { }


        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override PipelineType PipelineType => PipelineType.Flux;

        /// <summary>
        /// Gets the shift factor.
        /// </summary>
        protected override float ShiftFactor => 0.1159f;

        /// <summary>
        /// Gets the vae channel count.
        /// </summary>
        protected int VaeChannels => 16;

        /// <summary>
        /// Shoulds perform guidance.
        /// </summary>
        /// <param name="schedulerOptions">The scheduler options.</param>
        protected override bool ShouldPerformGuidance(SchedulerOptions schedulerOptions)
        {
            return schedulerOptions.GuidanceScale > 1;
        }

        /// <summary>
        /// Runs the scheduler steps.
        /// </summary>
        /// <param name="options"></param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async Task<DenseTensor<float>> DiffuseAsync(DiffuseOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var generateOptions = options.GenerateOptions;
            var schedulerOptions = generateOptions.SchedulerOptions;
            var performGuidance = ShouldPerformGuidance(schedulerOptions);
            var conditionalEmbeds = options.PromptEmbeddings.PromptEmbeds;
            var conditionalPooledEmbeds = options.PromptEmbeddings.PooledPromptEmbeds;
            var unconditionalEmbeds = options.PromptEmbeddings.NegativePromptEmbeds;
            var unconditionalPooledEmbeds = options.PromptEmbeddings.NegativePooledPromptEmbeds;

            var optimizations = GetOptimizations(generateOptions, options.PromptEmbeddings, progressCallback);
            using (var scheduler = GetScheduler(schedulerOptions))
            {
                // Get timesteps
                var timesteps = GetTimesteps(schedulerOptions, scheduler);

                // Create latent sample
                progressCallback.Notify("Prepare Input...");
                var latents = await PrepareLatentsAsync(generateOptions, scheduler, timesteps, cancellationToken);

                // Create ImageIds
                var imgIds = PrepareLatentImageIds(schedulerOptions);

                // Create TextIds
                var txtIds = PrepareLatentTextIds(conditionalEmbeds);

                // Get Model metadata
                var metadata = await _unet.LoadAsync(optimizations, cancellationToken);

                // Guiadance
                var guidanceTensor = new DenseTensor<float>(new float[] { schedulerOptions.GuidanceScale2 }, [1]);

                // Legacy Models
                if (metadata.Inputs[4].Dimensions.Length == 3)
                {
                    imgIds = imgIds.ReshapeTensor([1, .. imgIds.Dimensions]);
                    txtIds = txtIds.ReshapeTensor([1, .. txtIds.Dimensions]);
                }

                // Loop though the timesteps
                var step = 0;
                ReportProgress(progressCallback, "Step", 0, timesteps.Count, 0);
                foreach (var timestep in timesteps)
                {
                    step++;
                    var stepTime = Stopwatch.GetTimestamp();
                    cancellationToken.ThrowIfCancellationRequested();

                    // Create input tensor.
                    var inputTensor = scheduler.ScaleInput(latents, timestep);
                    var timestepTensor = CreateTimestepTensor(timestep);

                    // Transformer Inference
                    var conditionalResult = await RunTransformerAsync
                    (
                        metadata,
                        inputTensor,
                        timestepTensor,
                        conditionalEmbeds,
                        conditionalPooledEmbeds,
                        guidanceTensor,
                        imgIds,
                        txtIds,
                        cancellationToken
                    );

                    // Classifier free guidance
                    if (performGuidance)
                    {
                        // Transformer Inference
                        var unconditionalResult = await RunTransformerAsync
                        (
                            metadata,
                            inputTensor,
                            timestepTensor,
                            unconditionalEmbeds,
                            unconditionalPooledEmbeds,
                            guidanceTensor,
                            imgIds, txtIds,
                            cancellationToken
                        );
                        conditionalResult = PerformGuidance(conditionalResult, unconditionalResult, schedulerOptions.GuidanceScale);
                    }

                    // Scheduler Step
                    latents = scheduler.Step(conditionalResult, timestep, latents).Result;

                    ReportProgress(progressCallback, "Step", step, timesteps.Count, stepTime, UnpackLatents(latents, schedulerOptions.Width, schedulerOptions.Height));
                    _logger?.LogEnd(LogLevel.Debug, $"Step {step}/{timesteps.Count}", stepTime);
                }

                // Unload if required
                if (generateOptions.IsLowMemoryComputeEnabled)
                    await _unet.UnloadAsync();

                // Decode Latents
                return await DecodeLatentsAsync(generateOptions, UnpackLatents(latents, schedulerOptions.Width, schedulerOptions.Height), cancellationToken);
            }
        }


        /// <summary>
        /// Run transformer model inference
        /// </summary>
        /// <param name="transformerMetadata">The transformer metadata.</param>
        /// <param name="inputTensor">The input tensor.</param>
        /// <param name="timestepTensor">The timestep tensor.</param>
        /// <param name="promptEmbeds">The prompt embeds.</param>
        /// <param name="promptPooledEmbeds">The prompt pooled embeds.</param>
        /// <param name="guidanceTensor">The guidance tensor.</param>
        /// <param name="imgIds">The img ids.</param>
        /// <param name="txtIds">The text ids.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;DenseTensor`1&gt; representing the asynchronous operation.</returns>
        protected async Task<DenseTensor<float>> RunTransformerAsync(
            OnnxMetadata transformerMetadata,
            DenseTensor<float> inputTensor,
            DenseTensor<float> timestepTensor,
            DenseTensor<float> promptEmbeds,
            DenseTensor<float> promptPooledEmbeds,
            DenseTensor<float> guidanceTensor,
            DenseTensor<float> imgIds,
            DenseTensor<float> txtIds,
            CancellationToken cancellationToken)
        {
            using (var transformerParams = new OnnxInferenceParameters(transformerMetadata, cancellationToken))
            {
                transformerParams.AddInputTensor(inputTensor);
                transformerParams.AddInputTensor(promptEmbeds);
                transformerParams.AddInputTensor(promptPooledEmbeds);
                transformerParams.AddInputTensor(timestepTensor);
                transformerParams.AddInputTensor(imgIds);
                transformerParams.AddInputTensor(txtIds);
                if (transformerParams.InputCount == 7)
                    transformerParams.AddInputTensor(guidanceTensor);
                transformerParams.AddOutputBuffer(inputTensor.Dimensions);

                var transformerResults = await _unet.RunInferenceAsync(transformerParams);
                using (var transformerResult = transformerResults.FirstOrDefault())
                {
                    return transformerResult.ToDenseTensor();
                }
            }
        }


        /// <summary>
        /// Creates the timestep tensor.
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        protected override DenseTensor<float> CreateTimestepTensor(int timestep)
        {
            return new DenseTensor<float>(new float[] { timestep / 1000f }, [1]);
        }


        /// <summary>
        /// Prepares the latent image ids.
        /// </summary>
        /// <param name="latents">The latents.</param>
        /// <returns></returns>
        protected virtual DenseTensor<float> PrepareLatentImageIds(SchedulerOptions options)
        {
            var height = options.GetScaledHeight() / 2;
            var width = options.GetScaledWidth() / 2;
            var latentIds = new DenseTensor<float>([height, width, 3]);

            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                    latentIds[i, j, 1] += i;

            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                    latentIds[i, j, 2] += j;

            return latentIds.ReshapeTensor([latentIds.Dimensions[0] * latentIds.Dimensions[1], latentIds.Dimensions[2]]);
        }


        /// <summary>
        /// Prepares the latent text ids.
        /// </summary>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <param name="isLegacy">if set to <c>true</c> [is legacy].</param>
        /// <returns>DenseTensor&lt;System.Single&gt;.</returns>
        protected virtual DenseTensor<float> PrepareLatentTextIds(DenseTensor<float> promptEmbeds)
        {
            return new DenseTensor<float>([promptEmbeds.Dimensions[1], 3]);
        }


        /// <summary>
        /// Packs the latents.
        /// </summary>
        /// <param name="latents">The latents.</param>
        /// <returns></returns>
        protected DenseTensor<float> PackLatents(DenseTensor<float> latents)
        {
            var height = latents.Dimensions[2] / 2;
            var width = latents.Dimensions[3] / 2;
            latents = latents.ReshapeTensor([1, VaeChannels, height, 2, width, 2]);
            latents = latents.Permute([0, 2, 4, 1, 3, 5]);
            latents = latents.ReshapeTensor([1, height * width, VaeChannels * 4]);
            return latents;
        }


        /// <summary>
        /// Unpacks the latents.
        /// </summary>
        /// <param name="latents">The latents.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <returns></returns>
        protected DenseTensor<float> UnpackLatents(DenseTensor<float> latents, int width, int height)
        {
            var channels = latents.Dimensions[2];
            height = height / VaeChannels;
            width = width / VaeChannels;
            latents = latents.ReshapeTensor([1, height, width, channels / 4, 2, 2]);
            latents = latents.Permute([0, 3, 1, 4, 2, 5]);
            latents = latents.ReshapeTensor([1, channels / (2 * 2), height * 2, width * 2]);
            return latents;
        }


        /// <summary>
        /// Gets the scheduler.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="schedulerConfig">The scheduler configuration.</param>
        /// <returns></returns>
        protected override IScheduler GetScheduler(SchedulerOptions options)
        {
            return options.SchedulerType switch
            {
                SchedulerType.FlowMatchEulerDiscrete => new FlowMatchEulerDiscreteScheduler(options),
                SchedulerType.FlowMatchEulerDynamic => new FlowMatchEulerDynamicScheduler(options),
                _ => default
            };
        }


        /// <summary>
        /// Gets the optimizations.
        /// </summary>
        /// <param name="generateOptions">The generate options.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <returns>OnnxOptimizations.</returns>
        private OnnxOptimizations GetOptimizations(GenerateOptions generateOptions, PromptEmbeddingsResult promptEmbeddings, IProgress<DiffusionProgress> progressCallback = null)
        {
            var sampleSize = 64; // TODO: does this need to be in the UnetConfig?
            var optimizationLevel = generateOptions.OptimizationType == OptimizationType.None
                ? GraphOptimizationLevel.ORT_DISABLE_ALL
                : GraphOptimizationLevel.ORT_ENABLE_ALL;
            var optimizations = new OnnxOptimizations(optimizationLevel);

            // These are not added to the model graph but ensure model reloads on Width/Height
            optimizations.Add("dummy_width", generateOptions.SchedulerOptions.GetScaledWidth());
            optimizations.Add("dummy_height", generateOptions.SchedulerOptions.GetScaledHeight());
            if (generateOptions.OptimizationType >= OptimizationType.Level2)
            {
                optimizations.Add("transformer_sample_batch", 1);
                optimizations.Add("transformer_hidden_batch", 1);
                optimizations.Add("transformer_text_embeds_batch", 1);
                // optimizations.Add("time_batch", 1);
                optimizations.Add("transformer_img_ids_batch", 1);
                optimizations.Add("transformer_img_ids_length", 3);
                optimizations.Add("transformer_txt_ids_batch", 1);
                optimizations.Add("transformer_txt_ids_length", 3);
                optimizations.Add("transformer_sample_size", sampleSize);
                optimizations.Add("transformer_text_embeds_size", promptEmbeddings.PooledPromptEmbeds.Dimensions[1]);
                optimizations.Add("transformer_hidden_length", promptEmbeddings.PromptEmbeds.Dimensions[2]);
            }
            if (generateOptions.OptimizationType >= OptimizationType.Level3)
            {
                var sampleSequence = generateOptions.SchedulerOptions.Width * generateOptions.SchedulerOptions.Height / 256;
                optimizations.Add("transformer_sample_sequence", sampleSequence);
                optimizations.Add("transformer_img_ids_sequence", sampleSequence);
            }
            if (generateOptions.OptimizationType >= OptimizationType.Level4)
            {
                optimizations.Add("transformer_hidden_sequence", promptEmbeddings.PromptEmbeds.Dimensions[1]);
                optimizations.Add("transformer_txt_ids_sequence", promptEmbeddings.PromptEmbeds.Dimensions[1]);
            }

            if (_unet.HasOptimizationsChanged(optimizations))
            {
                progressCallback.Notify("Optimizing Pipeline...");
            }

            return optimizations;
        }

    }
}
