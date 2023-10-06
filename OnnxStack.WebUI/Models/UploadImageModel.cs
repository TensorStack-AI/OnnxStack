using System.ComponentModel.DataAnnotations;

namespace OnnxStack.Web.Models
{
    public class UploadImageModel
    {
        public int Width { get; set; }
        public int Height { get; set; }

        [Required]
        public string ImageBase64 { get; set; }
    }
}
