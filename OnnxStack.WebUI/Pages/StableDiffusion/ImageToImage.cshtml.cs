using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Models;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Config;
using Services;

namespace OnnxStack.WebUI.Pages.StableDiffusion
{
    public class ImageToImageModel : PageModel
    {
        private readonly ILogger<IndexModel> _logger;
        private readonly IFileService _fileService;

        public ImageToImageModel(ILogger<IndexModel> logger, IFileService fileService)
        {
            _logger = logger;
            _fileService = fileService;
        }

        [BindProperty]
        public PromptOptions Prompt { get; set; }

        [BindProperty]
        public SchedulerOptions Options { get; set; }

        [BindProperty]
        public InitialImageModel InitialImage { get; set; }

        public async Task OnGet(string img = null, int width = 0, int height = 0)
        {
            var fileResult = await _fileService.GetInputImageFile(img);
            if (fileResult is not null)
            {
                if (Constants.ValidSizes.Contains(width) && Constants.ValidSizes.Contains(height))
                    InitialImage = new InitialImageModel(fileResult.Filename, fileResult.FileUrl, width, height);
            }

            Prompt = new PromptOptions
            {
              //  Prompt = "A photo of a cat",
            };

            Options = new SchedulerOptions
            {
              //  InferenceSteps = 5,
            };
        }
    }
}