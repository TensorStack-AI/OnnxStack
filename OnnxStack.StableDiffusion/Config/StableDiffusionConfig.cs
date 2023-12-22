using OnnxStack.Common.Config;
using OnnxStack.Core;
using OnnxStack.Core.Config;
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
                modelSet.InitBlankTokenArray();
                foreach (var model in modelSet.ModelConfigurations)
                {
                    if ((model.Type == OnnxModelType.Tokenizer || model.Type == OnnxModelType.Tokenizer2) && string.IsNullOrEmpty(model.OnnxModelPath))
                        model.OnnxModelPath = defaultTokenizer;

                    if (!File.Exists(model.OnnxModelPath))
                        modelSet.IsEnabled = false;
                }
            }
        }
    }
}
