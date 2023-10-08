using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using OnnxStack.Web.Models;
using OnnxStack.WebUI.Models;
using Services;

namespace OnnxStack.WebUI.Pages.StableDiffusion
{
    public class IndexModel : PageModel
    {
        private readonly IFileService _fileService;
        private readonly ILogger<IndexModel> _logger;

        public IndexModel(ILogger<IndexModel> logger, IFileService fileService)
        {
            _logger = logger;
            _fileService = fileService;
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

        public async Task<ActionResult> OnPostUploadImage(UploadImageModel model)
        {
            if (!ModelState.IsValid)
                return Partial("UploadImageModal", model);

            var fileResult = await _fileService.UploadImageFile(model);
            if(fileResult is null)
            {
                ModelState.AddModelError("ImageBase64", "Error saving image file");
                return Partial("UploadImageModal", model);
            }

            return ModalResult.Close(new
            { 
                imageName = fileResult.Filename,
                imageUrl = fileResult.FileUrl,
                success = true
            });
        }
    }
}