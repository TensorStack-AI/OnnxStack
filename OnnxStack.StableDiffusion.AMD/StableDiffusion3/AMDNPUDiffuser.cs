using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers.StableDiffusion3;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.AMD.StableDiffusion3
{
    public abstract class AMDNPUDiffuser : StableDiffusion3Diffuser
    {
        protected AMDNPUDiffuser(UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, ILogger logger = null)
            : base(unet, vaeDecoder, vaeEncoder, logger)
        {
        }


        public override async Task<DenseTensor<float>> DiffuseAsync(DiffuseOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var performGuidance = true;
            var generateOptions = options.GenerateOptions;
            var schedulerOptions = generateOptions.SchedulerOptions;
            var promptEmbeddings = options.PromptEmbeddings.GetPromptEmbeds(performGuidance);
            var pooledEmbeddings = options.PromptEmbeddings.GetPooledPromptEmbeds(performGuidance);

            var optimizations = GetOptimizations(generateOptions, options.PromptEmbeddings, progressCallback);
            using (var scheduler = GetScheduler(schedulerOptions))
            {
                // Get Model metadata
                var metadata = await _unet.LoadAsync(optimizations, cancellationToken);

                // Get timesteps
                var timesteps = GetTimesteps(schedulerOptions, scheduler);

                // Create latent sample
                progressCallback.Notify("Prepare Input...");
                var latents = await PrepareLatentsAsync(generateOptions, scheduler, timesteps, cancellationToken);

                // Loop though the timesteps
                var step = 0;
                ReportProgress(progressCallback, "Step", 0, timesteps.Count, 0);
                foreach (var timestep in timesteps)
                {
                    step++;
                    var stepTime = Stopwatch.GetTimestamp();
                    cancellationToken.ThrowIfCancellationRequested();

                    // Create input tensor.
                    var inputTensor = scheduler.ScaleInput(latents.Repeat(2), timestep);
                    var timestepTensor = CreateTimestepTensor(timestep).Repeat(2);
                    var transformerOutputbuffer = schedulerOptions.GetScaledDimension(2, 16);

                    // Transformer Inference
                    var transformerResult = await RunTransformerAsync(metadata, inputTensor, timestepTensor, promptEmbeddings, pooledEmbeddings, transformerOutputbuffer, cancellationToken);

                    // Perform guidance
                    transformerResult = PerformGuidance(transformerResult, schedulerOptions.GuidanceScale);

                    // Scheduler Step
                    latents = scheduler.Step(transformerResult, timestep, latents).Result;

                    ReportProgress(progressCallback, "Step", step, timesteps.Count, stepTime, latents);
                    _logger?.LogEnd(LogLevel.Debug, $"Step {step}/{timesteps.Count}", stepTime);
                }

                // Unload if required
                if (generateOptions.IsLowMemoryComputeEnabled)
                    await _unet.UnloadAsync();


                // Decode Latents
                return await DecodeLatentsAsync(generateOptions, latents, cancellationToken);
            }
        }


        protected override OnnxOptimizations GetOptimizations(GenerateOptions generateOptions, PromptEmbeddingsResult promptEmbeddings, IProgress<DiffusionProgress> progressCallback = null)
        {
            var optimizationLevel = generateOptions.OptimizationType == OptimizationType.None
                ? GraphOptimizationLevel.ORT_DISABLE_ALL
                : GraphOptimizationLevel.ORT_ENABLE_ALL;
            var optimizations = new OnnxOptimizations(optimizationLevel);
            if (_unet.HasOptimizationsChanged(optimizations))
            {
                progressCallback.Notify("Optimizing Pipeline...");
            }
            return optimizations;
        }

    }
}
