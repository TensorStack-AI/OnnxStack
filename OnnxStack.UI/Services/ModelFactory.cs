using OnnxStack.Core.Config;
using OnnxStack.FeatureExtractor.Common;
using OnnxStack.ImageUpscaler.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.UI.Models;
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

        public IEnumerable<UpscaleModelTemplate> GetUpscaleModelTemplates()
        {
            yield return new UpscaleModelTemplate("Upscale 2x", 2, 512);
            yield return new UpscaleModelTemplate("Upscale 4x", 4, 512);
        }


        public IEnumerable<StableDiffusionModelTemplate> GetStableDiffusionModelTemplates()
        {
            yield return new StableDiffusionModelTemplate("SD", DiffuserPipelineType.StableDiffusion, ModelType.Base, 512, DiffuserType.TextToImage, DiffuserType.ImageToImage, DiffuserType.ImageInpaintLegacy);
            yield return new StableDiffusionModelTemplate("SD-Inpaint", DiffuserPipelineType.StableDiffusion, ModelType.Base, 512, DiffuserType.ImageInpaint);
            yield return new StableDiffusionModelTemplate("SD-ControlNet", DiffuserPipelineType.StableDiffusion, ModelType.Base, 512, DiffuserType.ControlNet, DiffuserType.ControlNetImage);

            yield return new StableDiffusionModelTemplate("SDXL", DiffuserPipelineType.StableDiffusionXL, ModelType.Base, 1024, DiffuserType.TextToImage, DiffuserType.ImageToImage, DiffuserType.ImageInpaintLegacy);
            yield return new StableDiffusionModelTemplate("SDXL-Inpaint", DiffuserPipelineType.StableDiffusionXL, ModelType.Base, 1024, DiffuserType.ImageInpaint);
            yield return new StableDiffusionModelTemplate("SDXL-Refiner", DiffuserPipelineType.StableDiffusionXL, ModelType.Refiner, 1024, DiffuserType.ImageToImage, DiffuserType.ImageInpaintLegacy);

            yield return new StableDiffusionModelTemplate("LCM", DiffuserPipelineType.LatentConsistency, ModelType.Base, 512, DiffuserType.TextToImage, DiffuserType.ImageToImage, DiffuserType.ImageInpaintLegacy);
            yield return new StableDiffusionModelTemplate("LCM-SDXL", DiffuserPipelineType.LatentConsistencyXL, ModelType.Base, 1024, DiffuserType.TextToImage, DiffuserType.ImageToImage, DiffuserType.ImageInpaintLegacy);

            yield return new StableDiffusionModelTemplate("InstaFlow", DiffuserPipelineType.InstaFlow, ModelType.Base, 512, DiffuserType.TextToImage);
        }


        public StableDiffusionModelSet CreateStableDiffusionModelSet(string name, string folder, StableDiffusionModelTemplate modelTemplate)
        {
            var modelSet = new StableDiffusionModelSet
            {
                Name = name,
                PipelineType = modelTemplate.PipelineType,
                Diffusers = modelTemplate.DiffuserTypes.ToList(),
                SampleSize = modelTemplate.SampleSize,
                DeviceId = _settings.DefaultDeviceId,
                ExecutionMode = _settings.DefaultExecutionMode,
                ExecutionProvider = _settings.DefaultExecutionProvider,
                InterOpNumThreads = _settings.DefaultInterOpNumThreads,
                IntraOpNumThreads = _settings.DefaultIntraOpNumThreads,
                MemoryMode = _settings.DefaultMemoryMode,
                IsEnabled = true,
            };

            // Some repositories have the ControlNet in the unet folder, some on the controlnet folder
            var isControlNet = modelTemplate.DiffuserTypes.Any(x => x == DiffuserType.ControlNet || x == DiffuserType.ControlNetImage);
            var unetPath = Path.Combine(folder, "unet", "model.onnx");
            var controlNetUnetPath = Path.Combine(folder, "controlnet", "model.onnx");
            if (isControlNet && File.Exists(controlNetUnetPath))
                unetPath = controlNetUnetPath;

            var tokenizerPath = Path.Combine(folder, "tokenizer", "model.onnx");
            var textEncoderPath = Path.Combine(folder, "text_encoder", "model.onnx");
            var vaeDecoder = Path.Combine(folder, "vae_decoder", "model.onnx");
            var vaeEncoder = Path.Combine(folder, "vae_encoder", "model.onnx");
            var tokenizer2Path = Path.Combine(folder, "tokenizer_2", "model.onnx");
            var textEncoder2Path = Path.Combine(folder, "text_encoder_2", "model.onnx");
            var controlnet = Path.Combine(folder, "controlnet", "model.onnx");
            if (!File.Exists(tokenizerPath))
                tokenizerPath = _defaultTokenizerPath;
            if (!File.Exists(tokenizer2Path))
                tokenizer2Path = _defaultTokenizerPath;

            if (modelSet.PipelineType == DiffuserPipelineType.StableDiffusionXL || modelSet.PipelineType == DiffuserPipelineType.LatentConsistencyXL)
            {
                if (modelTemplate.ModelType == ModelType.Refiner)
                {
                    modelSet.SampleSize = 1024;
                    modelSet.UnetConfig = new UNetConditionModelConfig { OnnxModelPath = unetPath, ModelType = ModelType.Refiner };
                    modelSet.Tokenizer2Config = new TokenizerModelConfig { OnnxModelPath = tokenizer2Path, TokenizerLength = 1280, PadTokenId = 1 };
                    modelSet.TextEncoder2Config = new TextEncoderModelConfig { OnnxModelPath = textEncoder2Path };
                    modelSet.VaeDecoderConfig = new AutoEncoderModelConfig { OnnxModelPath = vaeDecoder, ScaleFactor = 0.13025f };
                    modelSet.VaeEncoderConfig = new AutoEncoderModelConfig { OnnxModelPath = vaeEncoder, ScaleFactor = 0.13025f };
                }
                else
                {
                    modelSet.SampleSize = 1024;
                    modelSet.UnetConfig = new UNetConditionModelConfig { OnnxModelPath = unetPath, ModelType = ModelType.Base };
                    modelSet.TokenizerConfig = new TokenizerModelConfig { OnnxModelPath = tokenizerPath, PadTokenId = 1 };
                    modelSet.Tokenizer2Config = new TokenizerModelConfig { OnnxModelPath = tokenizer2Path, TokenizerLength = 1280, PadTokenId = 1 };
                    modelSet.TextEncoderConfig = new TextEncoderModelConfig { OnnxModelPath = textEncoderPath };
                    modelSet.TextEncoder2Config = new TextEncoderModelConfig { OnnxModelPath = textEncoder2Path };
                    modelSet.VaeDecoderConfig = new AutoEncoderModelConfig { OnnxModelPath = vaeDecoder, ScaleFactor = 0.13025f };
                    modelSet.VaeEncoderConfig = new AutoEncoderModelConfig { OnnxModelPath = vaeEncoder, ScaleFactor = 0.13025f };
                }
            }
            else
            {
                modelSet.SampleSize = 512;
                var tokenizerLength = modelTemplate.ModelType == ModelType.Turbo ? 1024 : 768;
                modelSet.UnetConfig = new UNetConditionModelConfig { OnnxModelPath = unetPath, ModelType = modelTemplate.ModelType };
                modelSet.TokenizerConfig = new TokenizerModelConfig { OnnxModelPath = tokenizerPath, TokenizerLength = tokenizerLength };
                modelSet.TextEncoderConfig = new TextEncoderModelConfig { OnnxModelPath = textEncoderPath };
                modelSet.VaeDecoderConfig = new AutoEncoderModelConfig { OnnxModelPath = vaeDecoder, ScaleFactor = 0.18215f };
                modelSet.VaeEncoderConfig = new AutoEncoderModelConfig { OnnxModelPath = vaeEncoder, ScaleFactor = 0.18215f };
            }

            return modelSet;
        }

        public UpscaleModelSet CreateUpscaleModelSet(string name, string filename, UpscaleModelTemplate modelTemplate)
        {
            return new UpscaleModelSet
            {
                Name = name,
                IsEnabled = true,
                DeviceId = _settings.DefaultDeviceId,
                ExecutionMode = _settings.DefaultExecutionMode,
                ExecutionProvider = _settings.DefaultExecutionProvider,
                InterOpNumThreads = _settings.DefaultInterOpNumThreads,
                IntraOpNumThreads = _settings.DefaultIntraOpNumThreads,
                UpscaleModelConfig = new UpscaleModelConfig
                {
                    Channels = 3,
                    SampleSize = modelTemplate.SampleSize,
                    ScaleFactor = modelTemplate.ScaleFactor,
                    OnnxModelPath = filename
                }
            };
        }


        public ControlNetModelSet CreateControlNetModelSet(string name, ControlNetType controlNetType, DiffuserPipelineType pipelineType, string modelFilename)
        {
            return new ControlNetModelSet
            {
                Name = name,
                Type = controlNetType,
                PipelineType = pipelineType,
                IsEnabled = true,
                DeviceId = _settings.DefaultDeviceId,
                ExecutionMode = _settings.DefaultExecutionMode,
                ExecutionProvider = _settings.DefaultExecutionProvider,
                InterOpNumThreads = _settings.DefaultInterOpNumThreads,
                IntraOpNumThreads = _settings.DefaultIntraOpNumThreads,
                ControlNetConfig = new ControlNetModelConfig
                {
                    OnnxModelPath = modelFilename
                }
            };
        }


        public FeatureExtractorModelSet CreateFeatureExtractorModelSet(string name, bool normalize, int sampleSize, int channels, string modelFilename)
        {
            return new FeatureExtractorModelSet
            {
                Name = name,
                IsEnabled = true,
                DeviceId = _settings.DefaultDeviceId,
                ExecutionMode = _settings.DefaultExecutionMode,
                ExecutionProvider = _settings.DefaultExecutionProvider,
                InterOpNumThreads = _settings.DefaultInterOpNumThreads,
                IntraOpNumThreads = _settings.DefaultIntraOpNumThreads,
                FeatureExtractorConfig = new FeatureExtractorModelConfig
                {
                    Channels = channels,
                    Normalize = normalize,
                    SampleSize = sampleSize,
                    OnnxModelPath = modelFilename
                }
            };
        }
    }
}
