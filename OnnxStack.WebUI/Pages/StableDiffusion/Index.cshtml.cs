using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using OnnxStack.Web.Models;

namespace OnnxStack.WebUI.Pages.StableDiffusion
{
    public class IndexModel : PageModel
    {
        private readonly ILogger<IndexModel> _logger;

        public IndexModel(ILogger<IndexModel> logger)
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

        public ActionResult OnPostUploadImage(UploadImageModel model)
        {
            if (!ModelState.IsValid)
                return Partial("UploadImageModal", model);

           //save base64 image to file
           System.IO.File.WriteAllBytes("image.png", Convert.FromBase64String(model.ImageBase64.Split(',')[1]));

            //return CloseModal(object);
            //return CloseModalError("Error Message");
            return ModalResult.Success();
        }
    }
}