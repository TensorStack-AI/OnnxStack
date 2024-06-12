using Microsoft.Extensions.Logging;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Diffusers.InstaFlow;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
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
        /// <param name="pipelineOptions">The pipeline options</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public InstaFlowPipeline(PipelineOptions pipelineOptions, TokenizerModel tokenizer, TextEncoderModel textEncoder, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, UNetConditionModel controlNet, List<DiffuserType> diffusers, List<SchedulerType> schedulers, SchedulerOptions defaultSchedulerOptions = default, ILogger logger = default)
            : base(pipelineOptions, tokenizer, textEncoder, unet, vaeDecoder, vaeEncoder, controlNet, diffusers, schedulers, defaultSchedulerOptions, logger)
        {
            _supportedDiffusers = diffusers ?? new List<DiffuserType>
            {
                DiffuserType.TextToImage
            };
            _supportedSchedulers = schedulers ?? new List<SchedulerType>
            {
                SchedulerType.InstaFlow
            };
            _defaultSchedulerOptions = defaultSchedulerOptions ?? new SchedulerOptions
            {
                InferenceSteps = 1,
                GuidanceScale = 0f,
                SchedulerType = SchedulerType.InstaFlow
            };
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
                DiffuserType.TextToImage => new TextDiffuser(_unet, _vaeDecoder, _vaeEncoder, _pipelineOptions.MemoryMode, _logger),
                DiffuserType.ControlNet => new ControlNetDiffuser(controlNetModel, _controlNetUnet, _vaeDecoder, _vaeEncoder, _pipelineOptions.MemoryMode, _logger),
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
            var config = modelSet with { };
            var unet = new UNetConditionModel(config.UnetConfig.ApplyDefaults(config));
            var tokenizer = new TokenizerModel(config.TokenizerConfig.ApplyDefaults(config));
            var textEncoder = new TextEncoderModel(config.TextEncoderConfig.ApplyDefaults(config));
            var vaeDecoder = new AutoEncoderModel(config.VaeDecoderConfig.ApplyDefaults(config));
            var vaeEncoder = new AutoEncoderModel(config.VaeEncoderConfig.ApplyDefaults(config));
            var controlnet = default(UNetConditionModel);
            if (config.ControlNetUnetConfig is not null)
                controlnet = new UNetConditionModel(config.ControlNetUnetConfig.ApplyDefaults(config));

            var pipelineOptions = new PipelineOptions(config.Name, config.MemoryMode);
            return new InstaFlowPipeline(pipelineOptions, tokenizer, textEncoder, unet, vaeDecoder, vaeEncoder, controlnet, config.Diffusers, config.Schedulers, config.SchedulerOptions, logger);
        }


        /// <summary>
        /// Creates the pipeline from a folder structure.
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static new InstaFlowPipeline CreatePipeline(string modelFolder, ModelType modelType = ModelType.Base, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML, MemoryModeType memoryMode = MemoryModeType.Maximum, ILogger logger = default)
        {
            return CreatePipeline(ModelFactory.CreateModelSet(modelFolder, DiffuserPipelineType.InstaFlow, modelType, deviceId, executionProvider, memoryMode), logger);
        }
    }
}
