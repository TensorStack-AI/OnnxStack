using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using OnnxStack.StableDiffusion.Config;

namespace OnnxStack.WebUI.Pages.StableDiffusion
{
    public class TextToImageModel : PageModel
    {
        private readonly ILogger<IndexModel> _logger;

        public TextToImageModel(ILogger<IndexModel> logger)
        {
            _logger = logger;
        }

        [BindProperty]
        public PromptOptions Prompt { get; set; }

        [BindProperty]
        public SchedulerOptions Options { get; set; }

        public void OnGet()
        {
            Prompt = new PromptOptions
            {
               // Prompt = "A photo of a cat",
            };

            Options = new SchedulerOptions
            {
              //  InferenceSteps = 5,
            };
        }
    }
}