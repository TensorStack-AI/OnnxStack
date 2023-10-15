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
        }

        public DateTime Timestamp { get; }
        public string Prompt { get; set; }
        public string NegativePrompt { get; set; }
        public SchedulerType SchedulerType { get; set; }
        public string MaskImageUrl { get; set; }
        public string InputImageUrl { get; set; }
        public string OutputImageUrl { get; set; }
        public SchedulerOptions SchedulerOptions { get; set; }
    }
}
