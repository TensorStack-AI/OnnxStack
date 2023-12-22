using OnnxStack.StableDiffusion.Config;
using System.Collections.Generic;

namespace OnnxStack.UI.Services
{
    public interface IModelFactory
    {
        IEnumerable<UpscaleModelTemplate> GetUpscaleModelTemplates();
        IEnumerable<StableDiffusionModelTemplate> GetStableDiffusionModelTemplates();

        UpscaleModelSet CreateUpscaleModelSet(string name, string filename, UpscaleModelTemplate modelTemplate);
        StableDiffusionModelSet CreateStableDiffusionModelSet(string name, string folder, StableDiffusionModelTemplate modelTemplate);
    }
}