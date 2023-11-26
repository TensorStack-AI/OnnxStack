using OnnxStack.StableDiffusion.Enums;
using System.Collections.Generic;

namespace OnnxStack.UI.Views
{
    public class ModelConfigTemplate
    {
        public string Name { get; set; }
        public string Description { get; set; }
        public string Author { get; set; }
        public string Repository { get; set; }
        public string ImageIcon { get; set; }
        public ModelTemplateStatus Status { get; set; }
        public int SampleSize { get; set; }
        public int PadTokenId { get; set; }
        public int BlankTokenId { get; set; }
        public int TokenizerLimit { get; set; }
        public bool IsDualTokenizer { get; set; }
        public int EmbeddingsLength { get; set; }
        public int DualEmbeddingsLength { get; set; }
        public float ScaleFactor { get; set; }
        public DiffuserPipelineType PipelineType { get; set; }
        public List<DiffuserType> Diffusers { get; set; } = new List<DiffuserType>();
        public List<string> ModelFiles { get; set; } = new List<string>();
        public List<string> Images { get; set; } = new List<string>();
    }

}
