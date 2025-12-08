using Microsoft.Extensions.Logging;
using OnnxStack.Core.Model;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.StableDiffusion.Pipelines;
using OnnxStack.StableDiffusion.Tokenizers;
using System;
using System.Collections.Generic;
using OnnxStack.StableDiffusion.Common;
using System.Threading.Tasks;
using System.Threading;
using System.Linq;

namespace OnnxStack.StableDiffusion.AMD.StableDiffusion
{
    public class AMDNPUStableDiffusionPipeline : StableDiffusionPipeline
    {
        public AMDNPUStableDiffusionPipeline(string name, ITokenizer tokenizer, TextEncoderModel textEncoder, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, UNetConditionModel controlNetUnet, List<DiffuserType> diffusers = null, List<SchedulerType> schedulers = null, SchedulerOptions defaultSchedulerOptions = null, ILogger logger = null)
            : base(name, tokenizer, textEncoder, unet, vaeDecoder, vaeEncoder, controlNetUnet, diffusers, schedulers, defaultSchedulerOptions, logger) { }


        protected override async Task<TokenizerResult> DecodePromptTextAsync(string inputText, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(inputText))
                return new TokenizerResult(Array.Empty<long>(), Array.Empty<long>());

            // NPU Model is fixed to 77 tokens, so truncate here
            var result = await _tokenizer.EncodeAsync(inputText);
            return result with
            {
                InputIds = result.InputIds.Take(_tokenizer.TokenizerLimit).ToArray(),
                AttentionMask = result.AttentionMask.Take(_tokenizer.TokenizerLimit).ToArray(),
            };
        }


        protected override IDiffuser CreateDiffuser(DiffuserType diffuserType, ControlNetModel controlNetModel)
        {
            return diffuserType switch
            {
                DiffuserType.TextToImage => new AMDNPUTextDiffuser(_unet, _vaeDecoder, _vaeEncoder, _logger),
                DiffuserType.ImageToImage => new AMDNPUImageDiffuser(_unet, _vaeDecoder, _vaeEncoder, _logger),
                DiffuserType.ImageInpaintLegacy => new AMDNPUImageDiffuser(_unet, _vaeDecoder, _vaeEncoder, _logger),
                DiffuserType.ControlNet => new AMDNPUControlNetDiffuser(controlNetModel, _controlNetUnet, _vaeDecoder, _vaeEncoder, _logger),
                DiffuserType.ControlNetImage => new AMDNPUControlNetImageDiffuser(controlNetModel, _controlNetUnet, _vaeDecoder, _vaeEncoder, _logger),
                _ => throw new NotImplementedException()
            };
        }


        public static new AMDNPUStableDiffusionPipeline CreatePipeline(StableDiffusionModelSet modelSet, ILogger logger = default)
        {
            var config = modelSet with { };
            var unet = new UNetConditionModel(config.UnetConfig.ApplyDefaults(config));
            var tokenizer = new ClipTokenizer(config.TokenizerConfig.ApplyDefaults(config));
            var textEncoder = new TextEncoderModel(config.TextEncoderConfig.ApplyDefaults(config));
            var vaeDecoder = new AutoEncoderModel(config.VaeDecoderConfig.ApplyDefaults(config));
            var vaeEncoder = new AutoEncoderModel(config.VaeEncoderConfig.ApplyDefaults(config));
            var controlnet = default(UNetConditionModel);
            if (config.ControlNetUnetConfig is not null)
                controlnet = new UNetConditionModel(config.ControlNetUnetConfig.ApplyDefaults(config));

            LogPipelineInfo(modelSet, logger);
            return new AMDNPUStableDiffusionPipeline(config.Name, tokenizer, textEncoder, unet, vaeDecoder, vaeEncoder, controlnet, config.Diffusers, config.Schedulers, config.SchedulerOptions, logger);
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
        public static new AMDNPUStableDiffusionPipeline CreatePipeline(OnnxExecutionProvider executionProvider, string modelFolder, ModelType modelType = ModelType.Base, ILogger logger = default)
        {
            return CreatePipeline(ModelFactory.CreateModelSet(modelFolder, PipelineType.StableDiffusion, modelType).WithProvider(executionProvider), logger);
        }
    }
}
