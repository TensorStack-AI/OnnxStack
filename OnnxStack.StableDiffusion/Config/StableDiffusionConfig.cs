using OnnxStack.Common.Config;
using OnnxStack.Core;
using System;
using System.Collections.Generic;
using System.IO;

namespace OnnxStack.StableDiffusion.Config
{
    public class StableDiffusionConfig : IConfigSection
    {
        public List<StableDiffusionModelSet> ModelSets { get; set; } = new List<StableDiffusionModelSet>();

        public void Initialize()
        {
            if (ModelSets.IsNullOrEmpty())
                return;

            var defaultTokenizer = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "cliptokenizer.onnx");
            if (!File.Exists(defaultTokenizer))
                defaultTokenizer = string.Empty;

            foreach (var modelSet in ModelSets)
            {
                if (modelSet.TokenizerConfig is not null)
                {
                    if (string.IsNullOrEmpty(modelSet.TokenizerConfig.OnnxModelPath) || !File.Exists(modelSet.TokenizerConfig.OnnxModelPath))
                        modelSet.TokenizerConfig.OnnxModelPath = defaultTokenizer;
                }

                if (modelSet.Tokenizer2Config is not null)
                {
                    if (string.IsNullOrEmpty(modelSet.Tokenizer2Config.OnnxModelPath) || !File.Exists(modelSet.Tokenizer2Config.OnnxModelPath))
                        modelSet.Tokenizer2Config.OnnxModelPath = defaultTokenizer;
                }
            }
        }
    }
}
