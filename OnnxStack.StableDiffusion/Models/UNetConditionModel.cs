using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Enums;

namespace OnnxStack.StableDiffusion.Models
{
    public class UNetConditionModel : OnnxModelSession
    {
        private readonly ModelType _modelType;
        public UNetConditionModel(UNetConditionModelConfig configuration) : base(configuration)
        {
            _modelType = configuration.ModelType;
        }

        public ModelType ModelType => _modelType;
    }


    public record UNetConditionModelConfig : OnnxModelConfig
    {
        public ModelType ModelType { get; set; }
    }
}
