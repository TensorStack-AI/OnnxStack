using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;

namespace OnnxStack.WebUI.Models
{
    public record ImageBlueprint
    {
        public ImageBlueprint(PromptOptions prompt, SchedulerOptions options, string outputImageUrl, string inputImageUrl = null, string maskImageUrl = null)
        {
            Timestamp = DateTime.Now;
            SchedulerOptions = options;
            MaskImageUrl = maskImageUrl;
            InputImageUrl = inputImageUrl;
            OutputImageUrl = outputImageUrl;
            Prompt = prompt.Prompt;
            NegativePrompt = prompt.NegativePrompt;
            SchedulerType = prompt.SchedulerType;
            DiffuserType = prompt.DiffuserType;
        }

        public DateTime Timestamp { get; }
        public string Prompt { get; }
        public string NegativePrompt { get; }
        public SchedulerType SchedulerType { get; }
        public DiffuserType DiffuserType { get; }
        public string MaskImageUrl { get; }
        public string InputImageUrl { get; }
        public string OutputImageUrl { get; }
        public SchedulerOptions SchedulerOptions { get; }
    }
}
