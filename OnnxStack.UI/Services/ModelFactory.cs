using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.UI.Models;
using OnnxStack.UI.Views;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace OnnxStack.UI.Services
{
    public class ModelFactory : IModelFactory
    {
        private readonly OnnxStackUIConfig _settings;
        private readonly string _defaultTokenizerPath;

        public ModelFactory(OnnxStackUIConfig settings)
        {
            _settings = settings;
            var defaultTokenizerPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "cliptokenizer.onnx");
            if (File.Exists(defaultTokenizerPath))
                _defaultTokenizerPath = defaultTokenizerPath;
        }

        public StableDiffusionModelSet CreateStableDiffusionModelSet(string name, string folder, string modelTemplateType)
        {
            var template = _settings.Templates
                .Where(x => x.Category == ModelTemplateCategory.StableDiffusion && x.Template == modelTemplateType && !x.IsUserTemplate)
                .FirstOrDefault();
            if (template == null)
                return null;

            return CreateStableDiffusionModelSet(name, folder, template.StableDiffusionTemplate);
        }

        public StableDiffusionModelSet CreateStableDiffusionModelSet(string name, string folder, StableDiffusionModelTemplate modelTemplate)
        {
            var modelSet = new StableDiffusionModelSet
            {
                Name = name,
                IsEnabled = true,
                SampleSize = modelTemplate.SampleSize,
                PipelineType = modelTemplate.PipelineType,
                Diffusers = modelTemplate.DiffuserTypes.ToList(),

                ScaleFactor = 0.18215f,
                TokenizerLimit = 77,
                PadTokenId = 49407,
                TokenizerLength = 768,
                Tokenizer2Length = 1280,
                BlankTokenId = 49407,
                TokenizerType = TokenizerType.One,
                ModelType = ModelType.Base,

                DeviceId = _settings.DefaultDeviceId,
                ExecutionMode = _settings.DefaultExecutionMode,
                ExecutionProvider = _settings.DefaultExecutionProvider,
                InterOpNumThreads = _settings.DefaultInterOpNumThreads,
                IntraOpNumThreads = _settings.DefaultIntraOpNumThreads,
                ModelConfigurations = new List<OnnxModelConfig>()
            };


            var unetPath = Path.Combine(folder, "unet", "model.onnx");
            var tokenizerPath = Path.Combine(folder, "tokenizer", "model.onnx");
            var textEncoderPath = Path.Combine(folder, "text_encoder", "model.onnx");
            var vaeDecoder = Path.Combine(folder, "vae_decoder", "model.onnx");
            var vaeEncoder = Path.Combine(folder, "vae_encoder", "model.onnx");
            var tokenizer2Path = Path.Combine(folder, "tokenizer_2", "model.onnx");
            var textEncoder2Path = Path.Combine(folder, "text_encoder_2", "model.onnx");
            if (!File.Exists(tokenizerPath))
                tokenizerPath = _defaultTokenizerPath;
            if (!File.Exists(tokenizer2Path))
                tokenizer2Path = _defaultTokenizerPath;

            if (modelSet.PipelineType == DiffuserPipelineType.StableDiffusionXL || modelSet.PipelineType == DiffuserPipelineType.LatentConsistencyXL)
            {
                modelSet.PadTokenId = 1;
                modelSet.ScaleFactor = 0.13025f;
                modelSet.TokenizerType = TokenizerType.Both;

                if (modelTemplate.ModelType == ModelType.Refiner)
                {
                    modelSet.ModelType = ModelType.Refiner;
                    modelSet.TokenizerType = TokenizerType.Two;
                    modelSet.Diffusers.Remove(DiffuserType.TextToImage);
                    modelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.Unet, OnnxModelPath = unetPath });
                    modelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.Tokenizer2, OnnxModelPath = tokenizer2Path });
                    modelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.TextEncoder2, OnnxModelPath = textEncoder2Path });
                    modelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.VaeDecoder, OnnxModelPath = vaeDecoder });
                    modelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.VaeEncoder, OnnxModelPath = vaeEncoder });
                }
                else
                {
                    modelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.Unet, OnnxModelPath = unetPath });
                    modelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.Tokenizer, OnnxModelPath = tokenizerPath });
                    modelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.Tokenizer2, OnnxModelPath = tokenizer2Path });
                    modelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.TextEncoder, OnnxModelPath = textEncoderPath });
                    modelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.TextEncoder2, OnnxModelPath = textEncoder2Path });
                    modelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.VaeDecoder, OnnxModelPath = vaeDecoder });
                    modelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.VaeEncoder, OnnxModelPath = vaeEncoder });
                }
            }
            else
            {
                modelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.Unet, OnnxModelPath = unetPath });
                modelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.Tokenizer, OnnxModelPath = tokenizerPath });
                modelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.TextEncoder, OnnxModelPath = textEncoderPath });
                modelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.VaeDecoder, OnnxModelPath = vaeDecoder });
                modelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.VaeEncoder, OnnxModelPath = vaeEncoder });
            }

            return modelSet;
        }

        public UpscaleModelSet CreateUpscaleModelSet(string name, string filename, string modelTemplateType)
        {
            var template = _settings.Templates
               .Where(x => x.Category == ModelTemplateCategory.StableDiffusion && x.Template == modelTemplateType && !x.IsUserTemplate)
               .FirstOrDefault();
            if (template == null)
                return null;

            return CreateUpscaleModelSet(name, filename, template.UpscaleTemplate);
        }


        public UpscaleModelSet CreateUpscaleModelSet(string name, string filename, UpscaleModelTemplate modelTemplate)
        {
            return new UpscaleModelSet
            {
                Name = name,
                IsEnabled = true,
                Channels = 3,
                SampleSize = modelTemplate.SampleSize,
                ScaleFactor = modelTemplate.ScaleFactor,
                ModelConfigurations = new List<OnnxModelConfig> { new OnnxModelConfig { Type = OnnxModelType.Upscaler, OnnxModelPath = filename } }
            };
        }
    }
}
