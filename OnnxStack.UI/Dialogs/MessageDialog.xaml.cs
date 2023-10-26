using OnnxStack.UI.Commands;
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
            OkCommand = new AsyncRelayCommand(Ok);
            NoCommand = new AsyncRelayCommand(No);
            YesCommand = new AsyncRelayCommand(Yes);
            InitializeComponent();
        }

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

        #region INotifyPropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;
        public void NotifyPropertyChanged([CallerMemberName] string property = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(property));
        }
        #endregion
    }
}
