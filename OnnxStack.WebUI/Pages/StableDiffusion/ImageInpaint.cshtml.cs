using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Models;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using Services;

namespace OnnxStack.WebUI.Pages.StableDiffusion
{
    public class ImageInpaintModel : PageModel
    {
        private readonly ILogger<IndexModel> _logger;
        private readonly IFileService _fileService;

        public ImageInpaintModel(ILogger<IndexModel> logger, IFileService fileService)
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
                    InitialImage = new InitialImageModel(fileResult.FileUrl, width, height);
            }

            Prompt = new PromptOptions
            {

            };

            Options = new SchedulerOptions
            {
                SchedulerType = SchedulerType.DDPM,
                Strength = 1.0f,
            };
        }
    }
}