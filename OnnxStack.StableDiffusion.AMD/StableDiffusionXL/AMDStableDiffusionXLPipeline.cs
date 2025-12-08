using Microsoft.Extensions.Logging;
using OnnxStack.Core;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.StableDiffusion.Pipelines;
using OnnxStack.StableDiffusion.Tokenizers;
using System.Collections.Generic;

namespace OnnxStack.StableDiffusion.AMD.StableDiffusionXL
{
    public class AMDStableDiffusionXLPipeline : StableDiffusionXLPipeline
    {
        public AMDStableDiffusionXLPipeline(string name, ITokenizer tokenizer, ITokenizer tokenizer2, TextEncoderModel textEncoder, TextEncoderModel textEncoder2, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, UNetConditionModel controlNet, List<DiffuserType> diffusers, List<SchedulerType> schedulers = null, SchedulerOptions defaultSchedulerOptions = null, ILogger logger = null)
        : base(name, tokenizer, tokenizer2, textEncoder, textEncoder2, unet, vaeDecoder, vaeEncoder, controlNet, diffusers, schedulers, defaultSchedulerOptions, logger)
        {
        }


        protected override IDiffuser CreateDiffuser(DiffuserType diffuserType, ControlNetModel controlNetModel)
        {
            if (diffuserType == DiffuserType.TextToImage)
                return new AMDTextDiffuser(_unet, _vaeDecoder, _vaeEncoder, _logger);

            return base.CreateDiffuser(diffuserType, controlNetModel);
        }


        public static new AMDStableDiffusionXLPipeline CreatePipeline(StableDiffusionModelSet modelSet, ILogger logger = default)
        {
            var config = modelSet with { };
            var unet = new UNetConditionModel(config.UnetConfig.ApplyDefaults(config));
            var tokenizer = new ClipTokenizer(config.TokenizerConfig.ApplyDefaults(config));
            var tokenizer2 = new ClipTokenizer(config.Tokenizer2Config.ApplyDefaults(config));
            var textEncoder = new TextEncoderModel(config.TextEncoderConfig.ApplyDefaults(config));
            var textEncoder2 = new TextEncoderModel(config.TextEncoder2Config.ApplyDefaults(config));
            var vaeDecoder = new AutoEncoderModel(config.VaeDecoderConfig.ApplyDefaults(config));
            var vaeEncoder = new AutoEncoderModel(config.VaeEncoderConfig.ApplyDefaults(config));
            var controlnet = default(UNetConditionModel);
            if (config.ControlNetUnetConfig is not null)
                controlnet = new UNetConditionModel(config.ControlNetUnetConfig.ApplyDefaults(config));

            LogPipelineInfo(modelSet, logger);
            return new AMDStableDiffusionXLPipeline(config.Name, tokenizer, tokenizer2, textEncoder, textEncoder2, unet, vaeDecoder, vaeEncoder, controlnet, config.Diffusers, config.Schedulers, config.SchedulerOptions, logger);
        }


        public static new AMDStableDiffusionXLPipeline CreatePipeline(OnnxExecutionProvider executionProvider, string modelFolder, ModelType modelType = ModelType.Base, ILogger logger = default)
        {
            return CreatePipeline(ModelFactory.CreateModelSet(modelFolder, PipelineType.StableDiffusionXL, modelType).WithProvider(executionProvider), logger);
        }

    }
}
