using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers.StableDiffusion;
using OnnxStack.StableDiffusion.Models;
using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.AMD.StableDiffusion
{
    public class AMDNPUControlNetDiffuser : ControlNetDiffuser
    {
        public AMDNPUControlNetDiffuser(ControlNetModel controlNet, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, ILogger logger = null)
        : base(controlNet, unet, vaeDecoder, vaeEncoder, logger) { }


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
            var promptEmbeds = options.PromptEmbeddings.GetPromptEmbeds(performGuidance);
            var pooledPromptEmbeds = options.PromptEmbeddings.GetPooledPromptEmbeds(performGuidance);
            using (var scheduler = GetScheduler(schedulerOptions))
            {
                // Get Model metadata
                var metadata = await _unet.LoadAsync(cancellationToken: cancellationToken);

                // Get Model metadata
                var controlNetMetadata = await _controlNet.LoadAsync(cancellationToken: cancellationToken);

                // Get timesteps
                var timesteps = GetTimesteps(schedulerOptions, scheduler);

                // Create latent sample
                progressCallback.Notify("Prepare Input...");
                var latents = await PrepareLatentsAsync(generateOptions, scheduler, timesteps, cancellationToken);

                // Control Image
                var controlImage = await PrepareControlImage(generateOptions);

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
                    var controlImageTensor = controlImage.Repeat(2);
                    var conditioningScale = CreateConditioningScaleTensor(schedulerOptions.ConditioningScale);

                    var outputDimension = schedulerOptions.GetScaledDimension(2);
                    using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
                    {
                        inferenceParameters.AddInputTensor(inputTensor);
                        inferenceParameters.AddInputTensor(timestepTensor);
                        inferenceParameters.AddInputTensor(promptEmbeds);

                        // ControlNet
                        using (var controlNetParameters = new OnnxInferenceParameters(controlNetMetadata, cancellationToken))
                        {
                            controlNetParameters.AddInputTensor(inputTensor);
                            controlNetParameters.AddInputTensor(timestepTensor);
                            controlNetParameters.AddInputTensor(promptEmbeds);
                            controlNetParameters.AddInputTensor(controlImageTensor);
                            if (controlNetMetadata.Inputs.Count == 5)
                                controlNetParameters.AddInputTensor(conditioningScale);

                            // Optimization: Pre-allocate device buffers for inputs
                            foreach (var item in controlNetMetadata.Outputs)
                                controlNetParameters.AddOutputBuffer();

                            // ControlNet inference
                            var controlNetResults = _controlNet.RunInference(controlNetParameters);

                            // Add ControlNet outputs to Unet input
                            foreach (var item in controlNetResults)
                                inferenceParameters.AddInput(item);

                            // Add output buffer
                            inferenceParameters.AddOutputBuffer(outputDimension);

                            // Unet inference
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
                    }

                    ReportProgress(progressCallback, "Step", step, timesteps.Count, stepTime, latents);
                    _logger?.LogEnd(LogLevel.Debug, $"Step {step}/{timesteps.Count}", stepTime);
                }

                // Unload if required
                if (generateOptions.IsLowMemoryComputeEnabled)
                    await Task.WhenAll(_controlNet.UnloadAsync(), _unet.UnloadAsync());

                // Decode Latents
                return await DecodeLatentsAsync(generateOptions, latents, cancellationToken);
            }
        }


        protected new DenseTensor<float> CreateTimestepTensor(int timestep)
        {
            // NPU Model batch 2
            return new DenseTensor<float>(new float[] { timestep, timestep }, [2]);
        }
    }
}
