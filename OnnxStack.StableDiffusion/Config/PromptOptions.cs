namespace OnnxStack.StableDiffusion.Config
{
    public class PromptOptions
    {
        public string Prompt { get; set; }
        public string NegativePrompt { get; set; }
        public SchedulerType SchedulerType { get; set; }
    }
}
