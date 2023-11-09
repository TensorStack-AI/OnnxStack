using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace OnnxStack.UI.Views
{
    /// <summary>
    /// Interaction logic for Logger.xaml
    /// </summary>
    public partial class Logger : UserControl, INavigatable, INotifyPropertyChanged
    {

        /// <summary>
        /// Initializes a new instance of the <see cref="Logger"/> class.
        /// </summary>
        public Logger()
        {
            ResetCommand = new AsyncRelayCommand(Reset, CanExecuteReset);
            InitializeComponent();
        }

        public OnnxStackUIConfig UISettings
        {
            get { return (OnnxStackUIConfig)GetValue(UISettingsProperty); }
            set { SetValue(UISettingsProperty, value); }
        }
        public static readonly DependencyProperty UISettingsProperty =
            DependencyProperty.Register("UISettings", typeof(OnnxStackUIConfig), typeof(Logger));

        public AsyncRelayCommand ResetCommand { get; }


        /// <summary>
        /// Gets or sets the log output.
        /// </summary>
        public string LogOutput
        {
            get { return (string)GetValue(LogOutputProperty); }
            set { SetValue(LogOutputProperty, value); }
        }
        public static readonly DependencyProperty LogOutputProperty =
            DependencyProperty.Register("LogOutput", typeof(string), typeof(Logger));


        public Task NavigateAsync(ImageResult imageResult)
        {
            throw new NotImplementedException();
        }


        /// <summary>
        /// Resets the log window.
        /// </summary>
        /// <returns></returns>
        private Task Reset()
        {
            LogOutput = null;
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can execute reset.
        /// </summary>
        /// <returns>
        ///   <c>true</c> if this instance can execute reset; otherwise, <c>false</c>.
        /// </returns>
        private bool CanExecuteReset()
        {
            return !string.IsNullOrEmpty(LogOutput);
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
