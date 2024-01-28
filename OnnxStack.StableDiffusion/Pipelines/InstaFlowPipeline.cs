using Microsoft.Extensions.Logging;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Diffusers.InstaFlow;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using System;
using System.Collections.Generic;

namespace OnnxStack.StableDiffusion.Pipelines
{
    public sealed class InstaFlowPipeline : StableDiffusionPipeline
    {

        /// <summary>
        /// Initializes a new instance of the <see cref="InstaFlowPipeline"/> class.
        /// </summary>
        /// <param name="name">The model name.</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public InstaFlowPipeline(string name, TokenizerModel tokenizer, TextEncoderModel textEncoder, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, List<DiffuserType> diffusers, ILogger logger = default)
            : base(name, tokenizer, textEncoder, unet, vaeDecoder, vaeEncoder, diffusers, logger)
        {
            _supportedDiffusers = diffusers ?? new List<DiffuserType> { DiffuserType.TextToImage };
            _supportedSchedulers = new List<SchedulerType> { SchedulerType.InstaFlow };
        }


        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override DiffuserPipelineType PipelineType => DiffuserPipelineType.InstaFlow;


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
                DiffuserType.ControlNet => new ControlNetDiffuser(controlNetModel, _unet, _vaeDecoder, _vaeEncoder, _logger),
                _ => throw new NotImplementedException()
            };
        }


        /// <summary>
        /// Creates the pipeline from a ModelSet configuration.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static new InstaFlowPipeline CreatePipeline(StableDiffusionModelSet modelSet, ILogger logger = default)
        {
            var unet = new UNetConditionModel(modelSet.UnetConfig.ApplyDefaults(modelSet));
            var tokenizer = new TokenizerModel(modelSet.TokenizerConfig.ApplyDefaults(modelSet));
            var textEncoder = new TextEncoderModel(modelSet.TextEncoderConfig.ApplyDefaults(modelSet));
            var vaeDecoder = new AutoEncoderModel(modelSet.VaeDecoderConfig.ApplyDefaults(modelSet));
            var vaeEncoder = new AutoEncoderModel(modelSet.VaeEncoderConfig.ApplyDefaults(modelSet));
            return new InstaFlowPipeline(modelSet.Name, tokenizer, textEncoder, unet, vaeDecoder, vaeEncoder, modelSet.Diffusers, logger);
        }
    }
}
