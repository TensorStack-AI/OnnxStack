using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.UI.Models;
using System.Collections.Generic;

namespace OnnxStack.UI.Services
{
    public interface IModelFactory
    {
        IEnumerable<UpscaleModelTemplate> GetUpscaleModelTemplates();
        IEnumerable<StableDiffusionModelTemplate> GetStableDiffusionModelTemplates();

        UpscaleModelSet CreateUpscaleModelSet(string name, string filename, UpscaleModelTemplate modelTemplate);
        StableDiffusionModelSet CreateStableDiffusionModelSet(string name, string folder, StableDiffusionModelTemplate modelTemplate);
        ControlNetModelSet CreateControlNetModelSet(string name, ControlNetType controlNetType, string modelFilename, string annotationFilename);
    }
}