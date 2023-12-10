using System.Windows.Media.Imaging;

namespace OnnxStack.UI.Models
{
    public record UpscaleResult(BitmapSource Image, UpscaleInfoModel Info, double Elapsed);
}
