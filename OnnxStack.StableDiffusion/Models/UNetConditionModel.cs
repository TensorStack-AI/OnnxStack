using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Enums;

namespace OnnxStack.StableDiffusion.Models
{
    public class UNetConditionModel : OnnxModelSession
    {
        private readonly UNetConditionModelConfig _configuration;

        public UNetConditionModel(UNetConditionModelConfig configuration) : base(configuration)
        {
            _configuration = configuration;
        }

        public ModelType ModelType => _configuration.ModelType;
    }


    public record UNetConditionModelConfig : OnnxModelConfig
    {
        public ModelType ModelType { get; set; }
    }
}
