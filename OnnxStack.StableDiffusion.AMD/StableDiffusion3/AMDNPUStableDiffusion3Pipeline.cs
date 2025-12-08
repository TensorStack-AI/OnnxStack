using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.StableDiffusion.Pipelines;
using OnnxStack.StableDiffusion.Tokenizers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.AMD.StableDiffusion3
{
    public class AMDNPUStableDiffusion3Pipeline : StableDiffusion3Pipeline
    {
        public AMDNPUStableDiffusion3Pipeline(string name, ITokenizer tokenizer, ITokenizer tokenizer2, ITokenizer tokenizer3, TextEncoderModel textEncoder, TextEncoderModel textEncoder2, TextEncoderModel textEncoder3, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, UNetConditionModel controlNet, List<DiffuserType> diffusers, List<SchedulerType> schedulers, SchedulerOptions defaultSchedulerOptions = null, ILogger logger = null) 
            : base(name, tokenizer, tokenizer2, tokenizer3, textEncoder, textEncoder2, textEncoder3, unet, vaeDecoder, vaeEncoder, controlNet, diffusers, schedulers, defaultSchedulerOptions, logger)
        {
        }


        protected override IDiffuser CreateDiffuser(DiffuserType diffuserType, ControlNetModel controlNetModel)
        {
            return diffuserType switch
            {
                DiffuserType.TextToImage => new AMDNPUTextDiffuser(_unet, _vaeDecoder, _vaeEncoder, _logger),
                DiffuserType.ImageToImage => new AMDNPUImageDiffuser(_unet, _vaeDecoder, _vaeEncoder, _logger),
                _ => throw new NotImplementedException()
            };
        }


        protected override async Task<TokenizerResult> DecodePromptTextAsync(string inputText, CancellationToken cancellationToken = default)
        {
            var truncatedLength = _tokenizer.TokenizerLimit; //TODO: 77 - should be dynamic
            var result = await base.DecodePromptTextAsync(inputText, cancellationToken);
            return new TokenizerResult
            (
               result.InputIds.Take(truncatedLength).ToArray(),
               result.AttentionMask.Take(truncatedLength).ToArray()
            );
        }


        protected override async Task<TokenizerResult> DecodeTextAsLongAsync(string inputText, CancellationToken cancellationToken = default)
        {
            var truncatedLength = _tokenizer2.TokenizerLimit; //TODO: 77 - should be dynamic
            var result = await base.DecodeTextAsLongAsync(inputText, cancellationToken);
            return new TokenizerResult
            (
               result.InputIds.Take(truncatedLength).ToArray(),
               result.AttentionMask.Take(truncatedLength).ToArray()
            );
        }


        protected override Task<DenseTensor<float>> GeneratePrompt3EmbedsAsync(TokenizerResult prompt3Tokens, int maxPromptTokenCount, CancellationToken cancellationToken = default)
        {
            var truncatedLength = 83; //TODO: 83? - should be 256
            prompt3Tokens = new TokenizerResult
            (
               prompt3Tokens.InputIds.Take(truncatedLength).ToArray(),
               prompt3Tokens.AttentionMask.Take(truncatedLength).ToArray()
            );
            return base.GeneratePrompt3EmbedsAsync(prompt3Tokens, truncatedLength, cancellationToken);
        }


        public static new AMDNPUStableDiffusion3Pipeline CreatePipeline(StableDiffusionModelSet modelSet, ILogger logger = default)
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
            var tokenizer3 = default(ITokenizer);
            if (config.Tokenizer3Config is not null)
                tokenizer3 = new SentencePieceTokenizer(config.Tokenizer3Config.ApplyDefaults(config));
            var textEncoder3 = default(TextEncoderModel);
            if (config.TextEncoder3Config is not null)
                textEncoder3 = new TextEncoderModel(config.TextEncoder3Config.ApplyDefaults(config));

            LogPipelineInfo(modelSet, logger);
            return new AMDNPUStableDiffusion3Pipeline(config.Name, tokenizer, tokenizer2, tokenizer3, textEncoder, textEncoder2, textEncoder3, unet, vaeDecoder, vaeEncoder, controlnet, config.Diffusers, config.Schedulers, config.SchedulerOptions, logger);
        }


        public static new AMDNPUStableDiffusion3Pipeline CreatePipeline(OnnxExecutionProvider executionProvider, string modelFolder, ModelType modelType = ModelType.Base, ILogger logger = default)
        {
            return CreatePipeline(ModelFactory.CreateModelSet(modelFolder, PipelineType.StableDiffusion3, modelType).WithProvider(executionProvider), logger);
        }

    }
}
