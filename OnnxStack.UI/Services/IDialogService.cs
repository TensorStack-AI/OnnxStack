using System.Windows;

namespace OnnxStack.UI.Services
{
    public interface IDialogService
    {
        T GetDialog<T>() where T : Window;
        T GetDialog<T>(Window owner) where T : Window;
    }

}
