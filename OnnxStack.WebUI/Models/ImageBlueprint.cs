using OnnxStack.StableDiffusion.Config;

namespace OnnxStack.WebUI.Models
{
    public record ImageBlueprint(PromptOptions Prompt, SchedulerOptions Options, string OutputImageUrl, string InputImageUrl = null)
    {
        public DateTime Timestamp { get; } = DateTime.UtcNow;
    }
}
