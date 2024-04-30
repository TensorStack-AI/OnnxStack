using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using System;
using System.Collections.Generic;
using System.IO;

namespace OnnxStack.StableDiffusion.Helpers
{
    public static class ModelFactory
    {
        public static string DefaultTokenizer => Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "cliptokenizer.onnx");


        /// <summary>
        /// Creates a StableDiffusionModelSet from the specified folder and pipeline type
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="pipeline">The pipeline.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns></returns>
        public static StableDiffusionModelSet CreateModelSet(string modelFolder, DiffuserPipelineType pipeline, ModelType modelType, int deviceId, ExecutionProvider executionProvider, MemoryModeType memoryMode)
        {
            if (pipeline == DiffuserPipelineType.StableCascade)
                return CreateStableCascadeModelSet(modelFolder, pipeline, modelType, deviceId, executionProvider, memoryMode);

            var tokenizerPath = Path.Combine(modelFolder, "tokenizer", "model.onnx");
            if (!File.Exists(tokenizerPath))
                tokenizerPath = DefaultTokenizer;

            var unetPath = Path.Combine(modelFolder, "unet", "model.onnx");
            var textEncoderPath = Path.Combine(modelFolder, "text_encoder", "model.onnx");
            var vaeDecoderPath = Path.Combine(modelFolder, "vae_decoder", "model.onnx");
            var vaeEncoderPath = Path.Combine(modelFolder, "vae_encoder", "model.onnx");
            var controlNetPath = Path.Combine(modelFolder, "controlNet", "model.onnx");

            // Some repositories have the ControlNet in the unet folder, some in the controlnet folder
            if (modelType == ModelType.ControlNet && File.Exists(controlNetPath))
                unetPath = controlNetPath;

            var diffusers = modelType switch
            {
                ModelType.Inpaint => new List<DiffuserType> { DiffuserType.ImageInpaint },
                ModelType.ControlNet => new List<DiffuserType> { DiffuserType.ControlNet, DiffuserType.ControlNetImage },
                _ => new List<DiffuserType> { DiffuserType.TextToImage, DiffuserType.ImageToImage, DiffuserType.ImageInpaintLegacy }
            };

            var sampleSize = 512;
            var tokenizer2Config = default(TokenizerModelConfig);
            var textEncoder2Config = default(TextEncoderModelConfig);
            var tokenizerConfig = new TokenizerModelConfig
            {
                OnnxModelPath = tokenizerPath
            };

            var textEncoderConfig = new TextEncoderModelConfig
            {
                OnnxModelPath = textEncoderPath
            };

            var unetConfig = new UNetConditionModelConfig
            {
                ModelType = modelType,
                OnnxModelPath = unetPath
            };

            var vaeDecoderConfig = new AutoEncoderModelConfig
            {
                ScaleFactor = 0.18215f,
                OnnxModelPath = vaeDecoderPath
            };

            var vaeEncoderConfig = new AutoEncoderModelConfig
            {
                ScaleFactor = 0.18215f,
                OnnxModelPath = vaeEncoderPath
            };

            // SDXL Pipelines
            if (pipeline == DiffuserPipelineType.StableDiffusionXL || pipeline == DiffuserPipelineType.LatentConsistencyXL)
            {
                // SDXL-Turbo has a 512 SampleSize
                if (modelType != ModelType.Turbo)
                    sampleSize = 1024;

                var textEncoder2Path = Path.Combine(modelFolder, "text_encoder_2", "model.onnx");
                var tokenizer2Path = Path.Combine(modelFolder, "tokenizer_2", "model.onnx");
                if (!File.Exists(tokenizer2Path))
                    tokenizer2Path = DefaultTokenizer;

                // Scale Factor
                vaeDecoderConfig.ScaleFactor = 0.13025f;
                vaeEncoderConfig.ScaleFactor = 0.13025f;

                tokenizerConfig.PadTokenId = 1;
                tokenizer2Config = new TokenizerModelConfig
                {
                    PadTokenId = 1,
                    TokenizerLength = 1280,
                    OnnxModelPath = tokenizer2Path
                };
                textEncoder2Config = new TextEncoderModelConfig
                {
                    OnnxModelPath = textEncoder2Path
                };
            }

            // SD-Turbo has TokenizerLength 1024
            if (pipeline == DiffuserPipelineType.StableDiffusion && modelType == ModelType.Turbo)
                tokenizerConfig.TokenizerLength = 1024;

            var configuration = new StableDiffusionModelSet
            {
                IsEnabled = true,
                SampleSize = sampleSize,
                Name = Path.GetFileNameWithoutExtension(modelFolder),
                PipelineType = pipeline,
                Diffusers = diffusers,
                DeviceId = deviceId,
                MemoryMode = memoryMode,
                ExecutionProvider = executionProvider,
                SchedulerOptions = GetDefaultSchedulerOptions(pipeline, modelType),
                TokenizerConfig = tokenizerConfig,
                Tokenizer2Config = tokenizer2Config,
                TextEncoderConfig = textEncoderConfig,
                TextEncoder2Config = textEncoder2Config,
                UnetConfig = unetConfig,
                VaeDecoderConfig = vaeDecoderConfig,
                VaeEncoderConfig = vaeEncoderConfig
            };
            return configuration;
        }

        public static StableDiffusionModelSet CreateStableCascadeModelSet(string modelFolder, DiffuserPipelineType pipeline, ModelType modelType, int deviceId, ExecutionProvider executionProvider, MemoryModeType memoryMode)
        {
            var tokenizerPath = Path.Combine(modelFolder, "tokenizer", "model.onnx");
            if (!File.Exists(tokenizerPath))
                tokenizerPath = DefaultTokenizer;

            var priorUnetPath = Path.Combine(modelFolder, "prior", "model.onnx");
            var decoderUnetPath = Path.Combine(modelFolder, "decoder", "model.onnx");
            var textEncoderPath = Path.Combine(modelFolder, "text_encoder", "model.onnx");
            var vqganPath = Path.Combine(modelFolder, "vqgan", "model.onnx");
            var imageEncoderPath = Path.Combine(modelFolder, "image_encoder", "model.onnx");

            var diffusers = new List<DiffuserType> { DiffuserType.TextToImage };

            var sampleSize = 1024;
            var tokenizer2Config = default(TokenizerModelConfig);
            var textEncoder2Config = default(TextEncoderModelConfig);

            var tokenizerConfig = new TokenizerModelConfig
            {
                TokenizerLength = 1280,
                OnnxModelPath = tokenizerPath
            };

            var textEncoderConfig = new TextEncoderModelConfig
            {
                OnnxModelPath = textEncoderPath
            };

            var priorUnetConfig = new UNetConditionModelConfig
            {
                OnnxModelPath = priorUnetPath
            };

            var decoderUnetConfig = new UNetConditionModelConfig
            {
                OnnxModelPath = decoderUnetPath
            };

            var vqganConfig = new AutoEncoderModelConfig
            {
                ScaleFactor = 0.3764f,
                OnnxModelPath = vqganPath
            };

            var imageEncoderConfig = new AutoEncoderModelConfig
            {
                OnnxModelPath = imageEncoderPath
            };


            var configuration = new StableDiffusionModelSet
            {
                IsEnabled = true,
                SampleSize = sampleSize,
                Name = Path.GetFileNameWithoutExtension(modelFolder),
                PipelineType = pipeline,
                Diffusers = diffusers,
                DeviceId = deviceId,
                MemoryMode = memoryMode,
                ExecutionProvider = executionProvider,
                SchedulerOptions = GetDefaultSchedulerOptions(pipeline, modelType),
                TokenizerConfig = tokenizerConfig,
                Tokenizer2Config = tokenizer2Config,
                TextEncoderConfig = textEncoderConfig,
                TextEncoder2Config = textEncoder2Config,
                UnetConfig = priorUnetConfig,
                DecoderUnetConfig = decoderUnetConfig,
                VaeDecoderConfig = vqganConfig,
                VaeEncoderConfig = imageEncoderConfig
            };
            return configuration;
        }

        /// <summary>
        /// Gets default scheduler options for specialized model types.
        /// </summary>
        /// <param name="pipelineType">Type of the pipeline.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <returns></returns>
        private static SchedulerOptions GetDefaultSchedulerOptions(DiffuserPipelineType pipelineType, ModelType modelType)
        {
            // SD-Turbo Models , 4 Steps, No Guidance
            if (pipelineType == DiffuserPipelineType.StableDiffusion && modelType == ModelType.Turbo)
                return new SchedulerOptions
                {
                    InferenceSteps = 4,
                    GuidanceScale = 0,
                    SchedulerType = SchedulerType.EulerAncestral
                };

            // SDXL-Turbo Models , 4 Steps, No Guidance, SampleSize: 512
            if (pipelineType == DiffuserPipelineType.StableDiffusionXL && modelType == ModelType.Turbo)
                return new SchedulerOptions
                {
                    Width = 512,
                    Height = 512,
                    InferenceSteps = 4,
                    GuidanceScale = 0,
                    SchedulerType = SchedulerType.EulerAncestral
                };

            return default;
        }
    }
}
