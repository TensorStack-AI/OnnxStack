using OnnxStack.StableDiffusion.Enums;

namespace OnnxStack.StableDiffusion.Config
{
    public record PipelineOptions(string Name, MemoryModeType MemoryMode);

}
