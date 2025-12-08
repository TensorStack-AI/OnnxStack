using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.StableDiffusion.Tokenizers;
using System.Collections.Generic;
using System.IO;

namespace OnnxStack.StableDiffusion.Helpers
{
    public static class ModelFactory
    {
        /// <summary>
        /// Creates a StableDiffusionModelSet from the specified folder and pipeline type
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="pipeline">The pipeline.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <returns></returns>
        public static StableDiffusionModelSet CreateModelSet(string modelFolder, PipelineType pipeline, ModelType modelType)
        {
            return pipeline switch
            {
                PipelineType.StableDiffusion2 => CreateStableDiffusion2ModelSet(modelFolder, modelType),
                PipelineType.StableDiffusionXL => CreateStableDiffusionXLModelSet(modelFolder, modelType),
                PipelineType.StableDiffusion3 => CreateStableDiffusion3ModelSet(modelFolder, modelType),
                PipelineType.StableCascade => CreateStableCascadeModelSet(modelFolder, modelType),
                PipelineType.Flux => CreateFluxModelSet(modelFolder, modelType),
                PipelineType.Locomotion => CreateLocomotionModelSet(modelFolder, 16),
                _ => CreateStableDiffusionModelSet(modelFolder, modelType, pipeline)
            };
        }


        /// <summary>
        /// Creates s StableDiffusion ModelSet.
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="pipeline">The pipeline.</param>
        public static StableDiffusionModelSet CreateStableDiffusionModelSet(string modelFolder, ModelType modelType, PipelineType pipeline)
        {
            var tokenizerPath = Path.Combine(modelFolder, "tokenizer", "vocab.json");
            var unetPath = Path.Combine(modelFolder, "unet", "model.onnx");
            var textEncoderPath = Path.Combine(modelFolder, "text_encoder", "model.onnx");
            var vaeDecoderPath = Path.Combine(modelFolder, "vae_decoder", "model.onnx");
            var vaeEncoderPath = Path.Combine(modelFolder, "vae_encoder", "model.onnx");
            var controlNetPath = Path.Combine(modelFolder, "controlnet", "model.onnx");

            var sampleSize = 512;
            var diffusers = new List<DiffuserType>
            {
                DiffuserType.TextToImage,
                DiffuserType.ImageToImage,
                DiffuserType.ImageInpaintLegacy
            };

            var tokenizerConfig = new TokenizerConfig
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

            var contronNetConfig = default(UNetConditionModelConfig);
            if (File.Exists(controlNetPath))
            {
                diffusers.Add(DiffuserType.ControlNetImage);
                if (modelType != ModelType.Instruct)
                    diffusers.Add(DiffuserType.ControlNet);

                contronNetConfig = new UNetConditionModelConfig
                {
                    ModelType = modelType,
                    OnnxModelPath = controlNetPath
                };
            }

            var configuration = new StableDiffusionModelSet
            {
                SampleSize = sampleSize,
                Name = Path.GetFileNameWithoutExtension(modelFolder),
                PipelineType = pipeline,
                Diffusers = diffusers,
                SchedulerOptions = GetDefaultSchedulerOptions(pipeline, modelType),
                TokenizerConfig = tokenizerConfig,
                TextEncoderConfig = textEncoderConfig,
                UnetConfig = unetConfig,
                VaeDecoderConfig = vaeDecoderConfig,
                VaeEncoderConfig = vaeEncoderConfig,
                ControlNetUnetConfig = contronNetConfig
            };
            return configuration;
        }


        /// <summary>
        /// Creates s StableDiffusion2 ModelSet.
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="pipeline">The pipeline.</param>
        public static StableDiffusionModelSet CreateStableDiffusion2ModelSet(string modelFolder, ModelType modelType)
        {
            var tokenizerPath = Path.Combine(modelFolder, "tokenizer", "vocab.json");
            var unetPath = Path.Combine(modelFolder, "unet", "model.onnx");
            var textEncoderPath = Path.Combine(modelFolder, "text_encoder", "model.onnx");
            var vaeDecoderPath = Path.Combine(modelFolder, "vae_decoder", "model.onnx");
            var vaeEncoderPath = Path.Combine(modelFolder, "vae_encoder", "model.onnx");
            var controlNetPath = Path.Combine(modelFolder, "controlnet", "model.onnx");

            var sampleSize = 768;
            var diffusers = new List<DiffuserType>
            {
                DiffuserType.TextToImage,
                DiffuserType.ImageToImage,
                DiffuserType.ImageInpaintLegacy
            };

            var tokenizerConfig = new TokenizerConfig
            {
                TokenizerLength = 1024,
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

            var contronNetConfig = default(UNetConditionModelConfig);
            if (File.Exists(controlNetPath))
            {
                diffusers.Add(DiffuserType.ControlNetImage);
                diffusers.Add(DiffuserType.ControlNet);
                contronNetConfig = new UNetConditionModelConfig
                {
                    ModelType = modelType,
                    OnnxModelPath = controlNetPath
                };
            }

            var configuration = new StableDiffusionModelSet
            {
                SampleSize = sampleSize,
                Name = Path.GetFileNameWithoutExtension(modelFolder),
                PipelineType = PipelineType.StableDiffusion2,
                Diffusers = diffusers,
                SchedulerOptions = GetDefaultSchedulerOptions(PipelineType.StableDiffusion2, modelType),
                TokenizerConfig = tokenizerConfig,
                TextEncoderConfig = textEncoderConfig,
                UnetConfig = unetConfig,
                VaeDecoderConfig = vaeDecoderConfig,
                VaeEncoderConfig = vaeEncoderConfig,
                ControlNetUnetConfig = contronNetConfig
            };
            return configuration;
        }


        /// <summary>
        /// Creates s StableDiffusionXL ModelSet.
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="pipeline">The pipeline.</param>
        public static StableDiffusionModelSet CreateStableDiffusionXLModelSet(string modelFolder, ModelType modelType)
        {
            var tokenizerPath = Path.Combine(modelFolder, "tokenizer", "vocab.json");
            var tokenizer2Path = Path.Combine(modelFolder, "tokenizer_2", "vocab.json");
            var textEncoderPath = Path.Combine(modelFolder, "text_encoder", "model.onnx");
            var textEncoder2Path = Path.Combine(modelFolder, "text_encoder_2", "model.onnx");
            var unetPath = Path.Combine(modelFolder, "unet", "model.onnx");
            var vaeDecoderPath = Path.Combine(modelFolder, "vae_decoder", "model.onnx");
            var vaeEncoderPath = Path.Combine(modelFolder, "vae_encoder", "model.onnx");
            var controlNetPath = Path.Combine(modelFolder, "controlnet", "model.onnx");

            var sampleSize = 1024;
            if (modelType == ModelType.Turbo)
                sampleSize = 512;

            var diffusers = new List<DiffuserType>
            {
                DiffuserType.TextToImage,
                DiffuserType.ImageToImage,
                DiffuserType.ImageInpaintLegacy
            };

            var tokenizerConfig = new TokenizerConfig
            {
                PadTokenId = 1,
                OnnxModelPath = tokenizerPath
            };

            var tokenizer2Config = new TokenizerConfig
            {
                PadTokenId = 1,
                TokenizerLength = 1280,
                OnnxModelPath = tokenizer2Path
            };

            var textEncoderConfig = new TextEncoderModelConfig
            {
                OnnxModelPath = textEncoderPath
            };

            var textEncoder2Config = new TextEncoderModelConfig
            {
                OnnxModelPath = textEncoder2Path
            };

            var unetConfig = new UNetConditionModelConfig
            {
                ModelType = modelType,
                OnnxModelPath = unetPath
            };

            var vaeDecoderConfig = new AutoEncoderModelConfig
            {
                ScaleFactor = 0.13025f,
                OnnxModelPath = vaeDecoderPath
            };

            var vaeEncoderConfig = new AutoEncoderModelConfig
            {
                ScaleFactor = 0.13025f,
                OnnxModelPath = vaeEncoderPath
            };

            var contronNetConfig = default(UNetConditionModelConfig);
            if (File.Exists(controlNetPath))
            {
                diffusers.Add(DiffuserType.ControlNetImage);
                diffusers.Add(DiffuserType.ControlNet);
                contronNetConfig = new UNetConditionModelConfig
                {
                    ModelType = modelType,
                    OnnxModelPath = controlNetPath
                };
            }

            var configuration = new StableDiffusionModelSet
            {
                SampleSize = sampleSize,
                Name = Path.GetFileNameWithoutExtension(modelFolder),
                PipelineType = PipelineType.StableDiffusionXL,
                Diffusers = diffusers,
                SchedulerOptions = GetDefaultSchedulerOptions(PipelineType.StableDiffusionXL, modelType),
                TokenizerConfig = tokenizerConfig,
                Tokenizer2Config = tokenizer2Config,
                TextEncoderConfig = textEncoderConfig,
                TextEncoder2Config = textEncoder2Config,
                UnetConfig = unetConfig,
                VaeDecoderConfig = vaeDecoderConfig,
                VaeEncoderConfig = vaeEncoderConfig,
                ControlNetUnetConfig = contronNetConfig
            };
            return configuration;
        }


        /// <summary>
        /// Creates s StableDiffusion3 ModelSet.
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="pipeline">The pipeline.</param>
        public static StableDiffusionModelSet CreateStableDiffusion3ModelSet(string modelFolder, ModelType modelType)
        {
            var tokenizerPath = Path.Combine(modelFolder, "tokenizer", "vocab.json");
            var tokenizer2Path = Path.Combine(modelFolder, "tokenizer_2", "vocab.json");
            var tokenizer3Path = Path.Combine(modelFolder, "tokenizer_3", "spiece.model");
            var textEncoderPath = Path.Combine(modelFolder, "text_encoder", "model.onnx");
            var textEncoder2Path = Path.Combine(modelFolder, "text_encoder_2", "model.onnx");
            var textEncoder3Path = Path.Combine(modelFolder, "text_encoder_3", "model.onnx");
            var transformerPath = GetTransformerPath(modelFolder);
            var vaeDecoderPath = Path.Combine(modelFolder, "vae_decoder", "model.onnx");
            var vaeEncoderPath = Path.Combine(modelFolder, "vae_encoder", "model.onnx");
            var controlNetPath = Path.Combine(modelFolder, "controlnet", "model.onnx");

            var sampleSize = 1024;
            var diffusers = new List<DiffuserType>
            {
                DiffuserType.TextToImage,
                DiffuserType.ImageToImage
            };

            var tokenizerConfig = new TokenizerConfig
            {
                PadTokenId = 1,
                OnnxModelPath = tokenizerPath
            };

            var tokenizerConfig2 = new TokenizerConfig
            {
                PadTokenId = 1,
                TokenizerLength = 1280,
                OnnxModelPath = tokenizer2Path
            };

            var tokenizer3Config = default(TokenizerConfig);
            if (File.Exists(tokenizer3Path))
            {
                tokenizer3Config = new TokenizerConfig
                {
                    PadTokenId = 1,
                    TokenizerLimit = 256,
                    TokenizerLength = 4096,
                    OnnxModelPath = tokenizer3Path
                };
            }

            var textEncoderConfig = new TextEncoderModelConfig
            {
                OnnxModelPath = textEncoderPath
            };

            var textEncoder2Config = new TextEncoderModelConfig
            {
                OnnxModelPath = textEncoder2Path
            };

            var textEncoder3Config = default(TextEncoderModelConfig);
            if (File.Exists(textEncoder3Path))
            {
                textEncoder3Config = new TextEncoderModelConfig
                {
                    OnnxModelPath = textEncoder3Path
                };
            }

            var transformerConfig = new UNetConditionModelConfig
            {
                ModelType = modelType,
                OnnxModelPath = transformerPath
            };

            var vaeDecoderConfig = new AutoEncoderModelConfig
            {
                ScaleFactor = 1.5305f,
                OnnxModelPath = vaeDecoderPath
            };

            var vaeEncoderConfig = new AutoEncoderModelConfig
            {
                ScaleFactor = 1.5305f,
                OnnxModelPath = vaeEncoderPath
            };

            var contronNetConfig = default(UNetConditionModelConfig);
            if (File.Exists(controlNetPath))
            {
                diffusers.Add(DiffuserType.ControlNet);
                diffusers.Add(DiffuserType.ControlNetImage);

                contronNetConfig = new UNetConditionModelConfig
                {
                    ModelType = modelType,
                    OnnxModelPath = controlNetPath
                };
            }

            var configuration = new StableDiffusionModelSet
            {
                SampleSize = sampleSize,
                Name = Path.GetFileNameWithoutExtension(modelFolder),
                PipelineType = PipelineType.StableDiffusion3,
                Diffusers = diffusers,
                SchedulerOptions = GetDefaultSchedulerOptions(PipelineType.StableDiffusion3, modelType),
                TokenizerConfig = tokenizerConfig,
                Tokenizer2Config = tokenizerConfig2,
                Tokenizer3Config = tokenizer3Config,
                TextEncoderConfig = textEncoderConfig,
                TextEncoder2Config = textEncoder2Config,
                TextEncoder3Config = textEncoder3Config,
                UnetConfig = transformerConfig,
                VaeDecoderConfig = vaeDecoderConfig,
                VaeEncoderConfig = vaeEncoderConfig,
                ControlNetUnetConfig = contronNetConfig
            };
            return configuration;
        }


        /// <summary>
        /// Creates the StableCascade ModelSet.
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        public static StableDiffusionModelSet CreateStableCascadeModelSet(string modelFolder, ModelType modelType)
        {
            var tokenizerPath = Path.Combine(modelFolder, "tokenizer", "vocab.json");
            var textEncoderPath = Path.Combine(modelFolder, "text_encoder", "model.onnx");
            var vaeDecoderPath = Path.Combine(modelFolder, "vae_decoder", "model.onnx");
            var vaeEncoderPath = Path.Combine(modelFolder, "vae_encoder", "model.onnx");
            var priorUnetPath = Path.Combine(modelFolder, "prior", "model.onnx");
            var decoderUnetPath = Path.Combine(modelFolder, "decoder", "model.onnx");

            var sampleSize = 1024;
            var diffusers = new List<DiffuserType>
            {
                DiffuserType.TextToImage
            };

            var tokenizerConfig = new TokenizerConfig
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

            var vaeDecoderConfig = new AutoEncoderModelConfig
            {
                ScaleFactor = 0.3764f,
                OnnxModelPath = vaeDecoderPath
            };

            var vaeEncoderConfig = new AutoEncoderModelConfig
            {
                OnnxModelPath = vaeEncoderPath
            };

            var configuration = new StableDiffusionModelSet
            {
                SampleSize = sampleSize,
                Name = Path.GetFileNameWithoutExtension(modelFolder),
                PipelineType = PipelineType.StableCascade,
                Diffusers = diffusers,
                SchedulerOptions = GetDefaultSchedulerOptions(PipelineType.StableCascade, modelType),
                TokenizerConfig = tokenizerConfig,
                TextEncoderConfig = textEncoderConfig,
                UnetConfig = priorUnetConfig,
                Unet2Config = decoderUnetConfig,
                VaeDecoderConfig = vaeDecoderConfig,
                VaeEncoderConfig = vaeEncoderConfig
            };
            return configuration;
        }


        /// <summary>
        /// Creates the Flux ModelSet.
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="maxSequenceLength">Maximum length of the sequence.</param>
        /// <param name="modelType">Type of the model.</param>
        public static StableDiffusionModelSet CreateFluxModelSet(string modelFolder, ModelType modelType)
        {
            var tokenizerPath = Path.Combine(modelFolder, "tokenizer", "vocab.json");
            var tokenizer2Path = Path.Combine(modelFolder, "tokenizer_2", "spiece.model");
            var textEncoderPath = Path.Combine(modelFolder, "text_encoder", "model.onnx");
            var textEncoder2Path = Path.Combine(modelFolder, "text_encoder_2", "model.onnx");
            var transformerPath = GetTransformerPath(modelFolder);
            var vaeDecoderPath = Path.Combine(modelFolder, "vae_decoder", "model.onnx");
            var vaeEncoderPath = Path.Combine(modelFolder, "vae_encoder", "model.onnx");

            var sampleSize = 1024;
            var maxSequenceLength = modelType == ModelType.Turbo ? 256 : 512;
            var diffusers = new List<DiffuserType>
            {
                DiffuserType.TextToImage,
                DiffuserType.ImageToImage
            };

            if (modelType == ModelType.Instruct)
                diffusers.Remove(DiffuserType.TextToImage);

            var tokenizerConfig = new TokenizerConfig
            {
                TokenizerLength = 768,
                OnnxModelPath = tokenizerPath
            };

            var tokenizer2Config = new TokenizerConfig
            {
                TokenizerLength = 4096,
                TokenizerLimit = maxSequenceLength,
                OnnxModelPath = tokenizer2Path
            };

            var textEncoderConfig = new TextEncoderModelConfig
            {
                OnnxModelPath = textEncoderPath
            };

            var textEncoder2Config = new TextEncoderModelConfig
            {
                OnnxModelPath = textEncoder2Path
            };

            var transformerConfig = new UNetConditionModelConfig
            {
                ModelType = modelType,
                OnnxModelPath = transformerPath
            };

            var vaeDecoderConfig = new AutoEncoderModelConfig
            {
                ScaleFactor = 0.3611f,
                OnnxModelPath = vaeDecoderPath
            };

            var vaeEncoderConfig = new AutoEncoderModelConfig
            {
                ScaleFactor = 0.3611f,
                OnnxModelPath = vaeEncoderPath
            };

            var configuration = new StableDiffusionModelSet
            {
                SampleSize = sampleSize,
                Name = Path.GetFileNameWithoutExtension(modelFolder),
                PipelineType = PipelineType.Flux,
                Diffusers = diffusers,
                SchedulerOptions = GetDefaultSchedulerOptions(PipelineType.Flux, modelType),
                TokenizerConfig = tokenizerConfig,
                Tokenizer2Config = tokenizer2Config,
                TextEncoderConfig = textEncoderConfig,
                TextEncoder2Config = textEncoder2Config,
                UnetConfig = transformerConfig,
                VaeDecoderConfig = vaeDecoderConfig,
                VaeEncoderConfig = vaeEncoderConfig
            };
            return configuration;
        }


        /// <summary>
        /// Creates the Locomotion odelSet.
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="contextSize">Size of the context.</param>
        public static StableDiffusionModelSet CreateLocomotionModelSet(string modelFolder, int contextSize)
        {
            var tokenizerPath = Path.Combine(modelFolder, "tokenizer", "vocab.json");
            var unetPath = Path.Combine(modelFolder, "unet", "model.onnx");
            var textEncoderPath = Path.Combine(modelFolder, "text_encoder", "model.onnx");
            var vaeDecoderPath = Path.Combine(modelFolder, "vae_decoder", "model.onnx");
            var vaeEncoderPath = Path.Combine(modelFolder, "vae_encoder", "model.onnx");
            var controlNetPath = Path.Combine(modelFolder, "controlnet", "model.onnx");
            var flowEstimationPath = Path.Combine(modelFolder, "flow_estimation", "model.onnx");
            var resamplePath = Path.Combine(modelFolder, "resample", "model.onnx");

            var sampleSize = 512;
            var diffusers = new List<DiffuserType>
            {
                DiffuserType.TextToVideo,
                DiffuserType.ImageToVideo,
                DiffuserType.VideoToVideo
            };

            var tokenizerConfig = new TokenizerConfig
            {
                TokenizerLimit = 77,
                OnnxModelPath = tokenizerPath
            };

            var textEncoderConfig = new TextEncoderModelConfig
            {
                OnnxModelPath = textEncoderPath
            };

            var unetConfig = new UNetConditionModelConfig
            {
                ContextSize = contextSize,
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

            var flowEstimationConfig = new FlowEstimationModelConfig
            {
                OnnxModelPath = flowEstimationPath
            };

            var resampleConfig = new ResampleModelConfig
            {
                OnnxModelPath = resamplePath
            };

            var contronNetConfig = default(UNetConditionModelConfig);
            if (File.Exists(controlNetPath))
            {
                diffusers.Add(DiffuserType.ControlNet);
                diffusers.Add(DiffuserType.ControlNetImage);
                diffusers.Add(DiffuserType.ControlNetVideo);
                contronNetConfig = new UNetConditionModelConfig
                {
                    ContextSize = contextSize,
                    OnnxModelPath = controlNetPath
                };
            }

            var configuration = new StableDiffusionModelSet
            {
                SampleSize = sampleSize,
                Name = Path.GetFileNameWithoutExtension(modelFolder),
                PipelineType = PipelineType.Locomotion,
                Diffusers = diffusers,
                SchedulerOptions = default,
                TokenizerConfig = tokenizerConfig,
                TextEncoderConfig = textEncoderConfig,
                UnetConfig = unetConfig,
                VaeDecoderConfig = vaeDecoderConfig,
                VaeEncoderConfig = vaeEncoderConfig,
                ControlNetUnetConfig = contronNetConfig,
                FlowEstimationConfig = flowEstimationConfig,
                ResampleModelConfig = resampleConfig
            };
            return configuration;
        }


        /// <summary>
        /// Gets default scheduler options for specialized model types.
        /// </summary>
        /// <param name="pipelineType">Type of the pipeline.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <returns></returns>
        private static SchedulerOptions GetDefaultSchedulerOptions(PipelineType pipelineType, ModelType modelType)
        {
            // SD-Turbo Models , 4 Steps, No Guidance
            if (pipelineType == PipelineType.StableDiffusion && modelType == ModelType.Turbo)
                return new SchedulerOptions
                {
                    InferenceSteps = 4,
                    GuidanceScale = 0,
                    SchedulerType = SchedulerType.EulerAncestral
                };

            // SDXL-Turbo Models , 4 Steps, No Guidance, SampleSize: 512
            if (pipelineType == PipelineType.StableDiffusionXL && modelType == ModelType.Turbo)
                return new SchedulerOptions
                {
                    Width = 512,
                    Height = 512,
                    InferenceSteps = 4,
                    GuidanceScale = 0,
                    SchedulerType = SchedulerType.EulerAncestral
                };

            if (pipelineType == PipelineType.StableDiffusion && modelType == ModelType.Instruct)
                return new SchedulerOptions
                {
                    Strength = 0.5f
                };

            // SD3-Turbo Models , 4 Steps, No Guidance
            if (pipelineType == PipelineType.StableDiffusion3 && modelType == ModelType.Turbo)
                return new SchedulerOptions
                {
                    Width = 1024,
                    Height = 1024,
                    InferenceSteps = 4,
                    GuidanceScale = 0,
                    Shift = 3,
                    SchedulerType = SchedulerType.FlowMatchEulerDiscrete
                };

            return default;
        }


        /// <summary>
        /// Gets the transformer path, "unet" or "transformer" folder.
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <returns>System.String.</returns>
        private static string GetTransformerPath(string modelFolder)
        {
            var transformerPath = Path.Combine(modelFolder, "transformer", "model.onnx");
            if (File.Exists(transformerPath))
                return transformerPath;

            return Path.Combine(modelFolder, "unet", "model.onnx");
        }


        public static StableDiffusionModelSet WithProvider(this StableDiffusionModelSet modelSet, OnnxExecutionProvider executionProvider)
        {
            modelSet.ExecutionProvider = executionProvider;
            return modelSet;
        }
    }
}
