using Microsoft.Extensions.Logging;
using OnnxStack.Core.Config;
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
    /// Interaction logic for SettingsView.xaml
    /// </summary>
    public partial class SettingsView : UserControl, INavigatable, INotifyPropertyChanged
    {
        private readonly ILogger<SettingsView> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="SettingsView"/> class.
        /// </summary>
        public SettingsView()
        {
            if (!DesignerProperties.GetIsInDesignMode(this))
                _logger = App.GetService<ILogger<SettingsView>>();

            SaveCommand = new AsyncRelayCommand(Save);
            InitializeComponent();
        }

        public AsyncRelayCommand SaveCommand { get; }

        public OnnxStackUIConfig UISettings
        {
            get { return (OnnxStackUIConfig)GetValue(UISettingsProperty); }
            set { SetValue(UISettingsProperty, value); }
        }
        public static readonly DependencyProperty UISettingsProperty =
            DependencyProperty.Register("UISettings", typeof(OnnxStackUIConfig), typeof(SettingsView));



        public Task NavigateAsync(ImageResult imageResult)
        {
            throw new NotImplementedException();
        }

        private Task Save()
        {
            try
            {
                ConfigManager.SaveConfiguration(UISettings);
            }
            catch (Exception ex)
            {
                _logger.LogError($"Error saving configuration file, {ex.Message}");
            }
            return Task.CompletedTask;
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
