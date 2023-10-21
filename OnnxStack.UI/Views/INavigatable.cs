using OnnxStack.UI.Models;
using System.Threading.Tasks;

namespace OnnxStack.UI.Views
{
    public interface INavigatable
    {
        Task NavigateAsync(ImageResult imageResult);
    }
}
