using OnnxStack.UI.Commands;
using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;

namespace OnnxStack.UI.Dialogs
{
    /// <summary>
    /// Interaction logic for MessageDialog.xaml
    /// </summary>
    public partial class MessageDialog : Window, INotifyPropertyChanged
    {
        private string _message;
        private MessageDialogType _dialogType;

        public MessageDialog()
        {
            WindowCloseCommand = new AsyncRelayCommand(WindowClose);
            WindowRestoreCommand = new AsyncRelayCommand(WindowRestore);
            WindowMinimizeCommand = new AsyncRelayCommand(WindowMinimize);
            WindowMaximizeCommand = new AsyncRelayCommand(WindowMaximize);
            OkCommand = new AsyncRelayCommand(Ok);
            NoCommand = new AsyncRelayCommand(No);
            YesCommand = new AsyncRelayCommand(Yes);
            InitializeComponent();
        }

        public AsyncRelayCommand WindowMinimizeCommand { get; }
        public AsyncRelayCommand WindowRestoreCommand { get; }
        public AsyncRelayCommand WindowMaximizeCommand { get; }
        public AsyncRelayCommand WindowCloseCommand { get; }
        public AsyncRelayCommand OkCommand { get; }
        public AsyncRelayCommand NoCommand { get; }
        public AsyncRelayCommand YesCommand { get; }
     
        public string Message
        {
            get { return _message; }
            set { _message = value; NotifyPropertyChanged(); }
        }
      
        public MessageDialogType DialogType
        {
            get { return _dialogType; }
            set { _dialogType = value; NotifyPropertyChanged(); }
        }

        public bool ShowDialog(string title, string message, MessageDialogType dialogType = MessageDialogType.Ok)
        {
            Title = title;
            Message = message;
            DialogType = dialogType;
            return ShowDialog() ?? false;
        }

        private Task Ok()
        {
            DialogResult = true;
            return Task.CompletedTask;
        }

        private Task No()
        {
            DialogResult = false;
            return Task.CompletedTask;
        }

        private Task Yes()
        {
            DialogResult = true;
            return Task.CompletedTask;
        }

        public enum MessageDialogType
        {
            Ok,
            YesNo
        }

        #region BaseWindow
        private Task WindowClose()
        {
            Close();
            return Task.CompletedTask;
        }

        private Task WindowRestore()
        {
            if (WindowState == WindowState.Maximized)
                WindowState = WindowState.Normal;
            else
                WindowState = WindowState.Maximized;
            return Task.CompletedTask;
        }

        private Task WindowMinimize()
        {
            WindowState = WindowState.Minimized;
            return Task.CompletedTask;
        }

        private Task WindowMaximize()
        {
            WindowState = WindowState.Maximized;
            return Task.CompletedTask;
        }

        private void OnContentRendered(object sender, EventArgs e)
        {
            InvalidateVisual();
        }
        #endregion

        #region INotifyPropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;
        public void NotifyPropertyChanged([CallerMemberName] string property = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(property));
        }
        #endregion
    }
}
