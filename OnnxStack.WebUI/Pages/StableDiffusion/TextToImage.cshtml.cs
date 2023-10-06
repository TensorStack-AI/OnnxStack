using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using OnnxStack.WebUI.Models;

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
        public TextToImageOptions Options { get; set; }

        public void OnGet()
        {
            Options = new TextToImageOptions
            {
                  Prompt = "photo of a cat"
            };
        }
    }
}