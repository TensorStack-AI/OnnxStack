using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using OnnxStack.Web.Models;
using Services;

namespace OnnxStack.WebUI.Pages.StableDiffusion
{
    public class IndexModel : PageModel
    {
        private readonly ILogger<IndexModel> _logger;

        public IndexModel(ILogger<IndexModel> logger, IFileService fileService)
        {
            _logger = logger;
        }

        public void OnGet()
        {
        }

        public ActionResult OnGetUploadImage(int width, int height)
        {
            return Partial("UploadImageModal", new UploadImageModel
            {
                Width = width,
                Height = height
            });
        }
    }
}