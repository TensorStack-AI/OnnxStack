using OnnxStack.Common.Config;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace OnnxStack.StableDiffusion.Config
{
    public class StableDiffusionConfig : IConfigSection
    {
        public List<ModelOptions> OnnxModelSets { get; set; } = new List<ModelOptions>();

        public void Initialize()
        {
            if (OnnxModelSets.IsNullOrEmpty())
                return;

            var defaultTokenizer = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "cliptokenizer.onnx");
            if (!File.Exists(defaultTokenizer))
                defaultTokenizer = string.Empty;

            foreach (var modelSet in OnnxModelSets)
            {
                modelSet.InitBlankTokenArray();
                foreach (var model in modelSet.ModelConfigurations.Where(x => x.Type == OnnxModelType.Tokenizer && string.IsNullOrEmpty(x.OnnxModelPath)))
                    model.OnnxModelPath = defaultTokenizer;
            }
        }
    }
}
