using OnnxStack.StableDiffusion.Config;
using OnnxStack.UI.Models;

namespace OnnxStack.UI.Services
{
    public interface IModelFactory
    {
        UpscaleModelSet CreateUpscaleModelSet(string name, string filename, string modelTemplateType);
        UpscaleModelSet CreateUpscaleModelSet(string name, string filename, UpscaleModelTemplate modelTemplate);
        StableDiffusionModelSet CreateStableDiffusionModelSet(string name, string folder, string modelTemplateType);
        StableDiffusionModelSet CreateStableDiffusionModelSet(string name, string folder, StableDiffusionModelTemplate modelTemplate);
    }
}