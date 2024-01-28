using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Diffusers.LatentConsistency;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Pipelines
{
    public sealed class LatentConsistencyPipeline : StableDiffusionPipeline
    {

        /// <summary>
        /// Initializes a new instance of the <see cref="LatentConsistencyPipeline"/> class.
        /// </summary>
        /// <param name="name">The model name.</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public LatentConsistencyPipeline(string name, TokenizerModel tokenizer, TextEncoderModel textEncoder, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, List<DiffuserType> diffusers, ILogger logger = default)
            : base(name, tokenizer, textEncoder, unet, vaeDecoder, vaeEncoder, diffusers, logger)
        {
            _supportedSchedulers = new List<SchedulerType> { SchedulerType.LCM };
        }


        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override DiffuserPipelineType PipelineType => DiffuserPipelineType.LatentConsistency;


        /// <summary>
        /// Runs the pipeline.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override Task<DenseTensor<float>> RunAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            // LCM does not support negative prompting
            promptOptions.NegativePrompt = string.Empty;
            return base.RunAsync(promptOptions, schedulerOptions, controlNet, progressCallback, cancellationToken);
        }


        /// <summary>
        /// Runs the pipeline batch.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="batchOptions">The batch options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override IAsyncEnumerable<BatchResult> RunBatchAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            // LCM does not support negative prompting
            promptOptions.NegativePrompt = string.Empty;
            return base.RunBatchAsync(promptOptions, schedulerOptions, batchOptions, controlNet, progressCallback, cancellationToken);
        }


        /// <summary>
        /// Check if we should run guidance.
        /// </summary>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <returns></returns>
        protected override bool ShouldPerformGuidance(SchedulerOptions schedulerOptions)
        {
            // LCM does not support Guidance
            return false;
        }


        /// <summary>
        /// Creates the diffuser.
        /// </summary>
        /// <param name="diffuserType">Type of the diffuser.</param>
        /// <param name="controlNetModel">The control net model.</param>
        /// <returns></returns>
        protected override IDiffuser CreateDiffuser(DiffuserType diffuserType, ControlNetModel controlNetModel)
        {
            return diffuserType switch
            {
                DiffuserType.TextToImage => new TextDiffuser(_unet, _vaeDecoder, _vaeEncoder, _logger),
                DiffuserType.ImageToImage => new ImageDiffuser(_unet, _vaeDecoder, _vaeEncoder, _logger),
                DiffuserType.ImageInpaintLegacy => new InpaintLegacyDiffuser(_unet, _vaeDecoder, _vaeEncoder, _logger),
                DiffuserType.ControlNet => new ControlNetDiffuser(controlNetModel, _unet, _vaeDecoder, _vaeEncoder, _logger),
                DiffuserType.ControlNetImage => new ControlNetImageDiffuser(controlNetModel, _unet, _vaeDecoder, _vaeEncoder, _logger),
                _ => throw new NotImplementedException()
            };
        }


        /// <summary>
        /// Creates the pipeline from a ModelSet configuration.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static new LatentConsistencyPipeline CreatePipeline(StableDiffusionModelSet modelSet, ILogger logger = default)
        {
            var unet = new UNetConditionModel(modelSet.UnetConfig.ApplyDefaults(modelSet));
            var tokenizer = new TokenizerModel(modelSet.TokenizerConfig.ApplyDefaults(modelSet));
            var textEncoder = new TextEncoderModel(modelSet.TextEncoderConfig.ApplyDefaults(modelSet));
            var vaeDecoder = new AutoEncoderModel(modelSet.VaeDecoderConfig.ApplyDefaults(modelSet));
            var vaeEncoder = new AutoEncoderModel(modelSet.VaeEncoderConfig.ApplyDefaults(modelSet));
            return new LatentConsistencyPipeline(modelSet.Name, tokenizer, textEncoder, unet, vaeDecoder, vaeEncoder, modelSet.Diffusers, logger);
        }
    }
}
