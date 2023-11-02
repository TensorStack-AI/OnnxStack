using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Config;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Schedulers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers.LatentConsistency
{
    public sealed class TextDiffuser : DiffuserBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TextDiffuser"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="onnxModelService">The onnx model service.</param>
        public TextDiffuser(IOnnxModelService onnxModelService, IPromptService promptService)
            : base(onnxModelService, promptService)
        {
        }

        public override async Task<DenseTensor<float>> DiffuseAsync(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default)
        {
            // Create random seed if none was set
            schedulerOptions.Seed = schedulerOptions.Seed > 0 ? schedulerOptions.Seed : Random.Shared.Next();

            // LCM does not support classifier-free guidance
            var guidance = schedulerOptions.GuidanceScale;
            schedulerOptions.GuidanceScale = 0f;

            // LCM does not support negative prompting
            promptOptions.NegativePrompt = string.Empty;

            // Get Scheduler
            using (var scheduler = GetScheduler(promptOptions, schedulerOptions))
            {
                // Process prompts
                var promptEmbeddings = await _promptService.CreatePromptAsync(modelOptions, promptOptions, schedulerOptions);

                // Get timesteps
                var timesteps = GetTimesteps(promptOptions, schedulerOptions, scheduler);

                // Create latent sample
                var latents = PrepareLatents(modelOptions, promptOptions, schedulerOptions, scheduler, timesteps);

                // Get Guidance Scale Embedding
                var guidanceEmbeddings = GetGuidanceScaleEmbedding(guidance);

                // Denoised result
                DenseTensor<float> denoised = null;

                // Loop though the timesteps
                var step = 0;
                foreach (var timestep in timesteps)
                {
                    step++;
                    cancellationToken.ThrowIfCancellationRequested();

                    // Create input tensor.
                    var inputTensor = scheduler.ScaleInput(latents, timestep);

                    // Create Input Parameters
                    var imputMeta = _onnxModelService.GetInputMetadata(modelOptions, OnnxModelType.Unet);
                    var inputNames = _onnxModelService.GetInputNames(modelOptions, OnnxModelType.Unet);
                    var inputParameters = CreateInputParameters(
                         NamedOnnxValue.CreateFromTensor(inputNames[0], inputTensor),
                         NamedOnnxValue.CreateFromTensor(inputNames[1], new DenseTensor<long>(new long[] { timestep }, new int[] { 1 })),
                         NamedOnnxValue.CreateFromTensor(inputNames[2], promptEmbeddings),
                         NamedOnnxValue.CreateFromTensor(inputNames[3], guidanceEmbeddings));

                    // Run Inference
                    using (var inferResult = await _onnxModelService.RunInferenceAsync(modelOptions, OnnxModelType.Unet, inputParameters))
                    {
                        var noisePred = inferResult.FirstElementAs<DenseTensor<float>>();

                        // Scheduler Step
                        var schedulerResult = scheduler.Step(noisePred, timestep, latents);

                        latents = schedulerResult.PreviousSample;
                        denoised = schedulerResult.ExtraSample;
                    }

                    progressCallback?.Invoke(step, timesteps.Count);
                }

                // Decode Latents
                return await DecodeLatents(modelOptions, promptOptions, schedulerOptions, denoised);
            }
        }


        /// <summary>
        /// Gets the timesteps.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <returns></returns>
        protected override IReadOnlyList<int> GetTimesteps(PromptOptions prompt, SchedulerOptions options, IScheduler scheduler)
        {
            return scheduler.Timesteps;
        }


        /// <summary>
        /// Prepares the latents for inference.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <returns></returns>
        protected override DenseTensor<float> PrepareLatents(IModelOptions model, PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps)
        {
            return scheduler.CreateRandomSample(options.GetScaledDimension(prompt.BatchCount), scheduler.InitNoiseSigma);
        }


        /// <summary>
        /// Gets the scheduler.
        /// </summary>
        /// <param name="prompt"></param>
        /// <param name="options">The options.</param>
        /// <returns></returns>
        protected override IScheduler GetScheduler(PromptOptions prompt, SchedulerOptions options)
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
        public DenseTensor<float> GetGuidanceScaleEmbedding(float guidance, int embeddingDim = 256)
        {
            // TODO:
            //assert len(w.shape) == 1
            //w = w * 1000.0

            //half_dim = embedding_dim // 2
            //emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            //emb = torch.exp(torch.arange(half_dim, dtype = dtype) * -emb)
            //emb = w.to(dtype)[:, None] * emb[None, :]
            //emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim = 1)
            //if embedding_dim % 2 == 1:  # zero pad
            //    emb = torch.nn.functional.pad(emb, (0, 1))
            //assert emb.shape == (w.shape[0], embedding_dim)
            //return emb

            var w = guidance - 1f;

            var half_dim = embeddingDim / 2;

            var log = MathF.Log(10000.0f) / (half_dim - 1);

            var emb = Enumerable.Range(0, half_dim)
                .Select(x => MathF.Exp(x * -log))
                .ToArray();
            var embSin = emb.Select(MathF.Sin).ToArray();
            var embCos = emb.Select(MathF.Cos).ToArray();

            DenseTensor<float> result = new DenseTensor<float>(new[] { 1, 2 * half_dim });
            for (int i = 0; i < half_dim; i++)
            {
                result[0, i] = embSin[i];
                result[0, i + half_dim] = embCos[i];
            }

            return result;
        }
    }
}
