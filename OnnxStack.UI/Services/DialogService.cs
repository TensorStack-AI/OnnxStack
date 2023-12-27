using System.Windows;

namespace OnnxStack.UI.Services
{
    public class DialogService : IDialogService
    {

        public T GetDialog<T>() where T : Window
        {
            return Resolve<T>(Application.Current.MainWindow);
        }

        public T GetDialog<T>(Window owner) where T : Window
        {
            return Resolve<T>(owner);
        }

        private T Resolve<T>(Window owner) where T : Window
        {
            var dialog = App.GetService<T>();
            dialog.Owner = owner;
            return dialog;
        }
    }
}
