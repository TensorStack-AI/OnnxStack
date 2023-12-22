using OnnxStack.StableDiffusion.Config;

namespace OnnxStack.UI.Services
{
    public interface IModelFactory
    {
        UpscaleModelSet CreateUpscaleModelSet(string name, string filename, UpscaleModelTemplate modelTemplate);
        StableDiffusionModelSet CreateStableDiffusionModelSet(string name, string folder, StableDiffusionModelTemplate modelTemplate);
    }
}