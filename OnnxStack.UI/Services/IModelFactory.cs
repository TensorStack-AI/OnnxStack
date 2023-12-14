using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;

namespace OnnxStack.UI.Services
{
    public interface IModelFactory
    {
        StableDiffusionModelSet CreateModelSet(string name, string folder, DiffuserPipelineType pipeline, ModelType modelType);
    }
}