using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Model;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers.StableDiffusion;
using OnnxStack.StableDiffusion.Models;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Threading;
using System;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using OnnxStack.StableDiffusion.Enums;

namespace OnnxStack.StableDiffusion.AMD.StableDiffusion
{
    public abstract class AMDNPUDiffuser : StableDiffusionDiffuser
    {
        protected AMDNPUDiffuser(UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, ILogger logger = null)
            : base(unet, vaeDecoder, vaeEncoder, logger) { }


        protected override bool ShouldPerformGuidance(SchedulerOptions schedulerOptions)
        {
            // NPU Model is fixed batch 2
            return true;
        }


        public override async Task<DenseTensor<float>> DiffuseAsync(DiffuseOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var generateOptions = options.GenerateOptions;
            var schedulerOptions = generateOptions.SchedulerOptions;
            var performGuidance = ShouldPerformGuidance(schedulerOptions);
            var promptEmbeddings = options.PromptEmbeddings.GetPromptEmbeds(performGuidance);

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
                    var inputLatent = latents.Repeat(2);
                    var inputTensor = scheduler.ScaleInput(inputLatent, timestep);
                    var timestepTensor = CreateTimestepTensor(timestep);

                    var outputDimension = schedulerOptions.GetScaledDimension(2);
                    using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
                    {
                        inferenceParameters.AddInputTensor(inputTensor);
                        inferenceParameters.AddInputTensor(timestepTensor);
                        inferenceParameters.AddInputTensor(promptEmbeddings);
                        inferenceParameters.AddOutputBuffer(outputDimension);

                        var results = await _unet.RunInferenceAsync(inferenceParameters);
                        using (var result = results.First())
                        {
                            var noisePred = result.ToDenseTensor();

                            // Perform guidance
                            noisePred = PerformGuidance(noisePred, schedulerOptions.GuidanceScale);

                            // Scheduler Step
                            latents = scheduler.Step(noisePred, timestep, latents).Result;
                        }
                    }

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


        protected new DenseTensor<double> CreateTimestepTensor(int timestep)
        {
            return new DenseTensor<double>(new double[] { timestep }, [1]);
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
