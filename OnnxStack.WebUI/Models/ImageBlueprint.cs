using OnnxStack.StableDiffusion.Config;

namespace OnnxStack.WebUI.Models
{
    public record ImageBlueprint(PromptOptions Prompt, SchedulerOptions Options)
    {
        public DateTime Timestamp { get; } = DateTime.UtcNow;
    }
}
