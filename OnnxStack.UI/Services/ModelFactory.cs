using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.UI.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime;
using System.Text;

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

        public StableDiffusionModelSet CreateModelSet(string name, string folder, DiffuserPipelineType pipeline, ModelType modelType)
        {
            var modelSet = new StableDiffusionModelSet
            {
                Name = name,
                PipelineType = pipeline,
                ScaleFactor = 0.18215f,
                TokenizerLimit = 77,
                PadTokenId = 49407,
                TokenizerLength = 768,
                Tokenizer2Length = 1280,
                BlankTokenId = 49407,
                Diffusers = Enum.GetValues<DiffuserType>().ToList(),
                SampleSize = 512,
                TokenizerType = TokenizerType.One,
                ModelType = ModelType.Base,

                DeviceId = _settings.DefaultDeviceId,
                ExecutionMode = _settings.DefaultExecutionMode,
                ExecutionProvider = _settings.DefaultExecutionProvider,
                InterOpNumThreads = _settings.DefaultInterOpNumThreads,
                IntraOpNumThreads = _settings.DefaultIntraOpNumThreads,
                IsEnabled = true,
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

            if (pipeline == DiffuserPipelineType.StableDiffusionXL || pipeline == DiffuserPipelineType.LatentConsistencyXL)
            {
                modelSet.PadTokenId = 1;
                modelSet.SampleSize = 1024;
                modelSet.ScaleFactor = 0.13025f;
                modelSet.TokenizerType = TokenizerType.Both;

                if (modelType == ModelType.Refiner)
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

    }
}
