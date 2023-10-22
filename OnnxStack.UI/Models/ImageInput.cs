using System.Text.Json.Serialization;
using System.Windows.Media.Imaging;

namespace OnnxStack.UI.Models
{
    public class ImageInput
    {
        [JsonIgnore]
        public BitmapSource Image { get; init; }
        public string FileName { get; set; }
        public string FileSize { get; set; }
    }
}