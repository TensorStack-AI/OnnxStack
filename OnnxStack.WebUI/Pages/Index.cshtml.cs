using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using OnnxStack.StableDiffusion.Config;

namespace OnnxStack.WebUI.Pages
{
    public class IndexModel : PageModel
    {
        private readonly ILogger<IndexModel> _logger;

        public IndexModel(ILogger<IndexModel> logger)
        {
            _logger = logger;
        }

        [BindProperty]
        public PromptOptions PromptOptions { get; set; }

        [BindProperty]
        public SchedulerOptions SchedulerOptions { get; set; }

        public void OnGet()
        {

        }
    }
}