using OnnxStack.StableDiffusion.Common;

namespace OnnxStack.StableDiffusion.Config
{
    public record DiffuseOptions(GenerateOptions GenerateOptions, PromptEmbeddingsResult PromptEmbeddings);
}
