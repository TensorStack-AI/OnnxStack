using Microsoft.Extensions.Logging;
using OnnxStack.Core;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
using OnnxStack.UI.Services;
using System;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace OnnxStack.UI.UserControls
{
    /// <summary>
    /// Interaction logic for FeaturePickerControl.xaml
    /// </summary>
    public partial class FeaturePickerControl : UserControl, INotifyPropertyChanged
    {
        private readonly ILogger<FeaturePickerControl> _logger;
        private readonly IFeatureExtractorService _featureExtractorService;

        /// <summary>Initializes a new instance of the <see cref="FeaturePickerControl" /> class.</summary>
        public FeaturePickerControl()
        {
            if (!DesignerProperties.GetIsInDesignMode(this))
            {
                _logger = App.GetService<ILogger<FeaturePickerControl>>();
                _featureExtractorService = App.GetService<IFeatureExtractorService>();
            }

            LoadCommand = new AsyncRelayCommand(LoadModel);
            UnloadCommand = new AsyncRelayCommand(UnloadModel);
            InitializeComponent();
        }

        public AsyncRelayCommand LoadCommand { get; set; }
        public AsyncRelayCommand UnloadCommand { get; set; }

        public OnnxStackUIConfig Settings
        {
            get { return (OnnxStackUIConfig)GetValue(SettingsProperty); }
            set { SetValue(SettingsProperty, value); }
        }
        public static readonly DependencyProperty SettingsProperty =
            DependencyProperty.Register("Settings", typeof(OnnxStackUIConfig), typeof(FeaturePickerControl));


        /// <summary>
        /// Gets or sets the selected model.
        /// </summary>
        public FeatureExtractorModelSetViewModel SelectedModel
        {
            get { return (FeatureExtractorModelSetViewModel)GetValue(SelectedModelProperty); }
            set { SetValue(SelectedModelProperty, value); }
        }
        public static readonly DependencyProperty SelectedModelProperty =
            DependencyProperty.Register("SelectedModel", typeof(FeatureExtractorModelSetViewModel), typeof(FeaturePickerControl));



        /// <summary>
        /// Loads the model.
        /// </summary>
        private async Task LoadModel()
        {
            if (_featureExtractorService.IsModelLoaded(SelectedModel.ModelSet))
                return;

            var elapsed = _logger.LogBegin($"'{SelectedModel.Name}' Loading...");
            SelectedModel.IsLoaded = false;
            SelectedModel.IsLoading = true;

            try
            {
                foreach (var model in Settings.FeatureExtractorModelSets.Where(x => x.IsLoaded))
                {
                    _logger.LogInformation($"'{model.Name}' Unloading...");
                    await _featureExtractorService.UnloadModelAsync(model.ModelSet);
                    model.IsLoaded = false;
                }
                SelectedModel.IsLoaded = await _featureExtractorService.LoadModelAsync(SelectedModel.ModelSet);
            }
            catch (Exception ex)
            {
                _logger.LogError($"An error occured while loading model '{SelectedModel.Name}' \n {ex}");
            }

            SelectedModel.IsLoading = false;
            _logger.LogEnd($"'{SelectedModel.Name}' Loaded.", elapsed);
        }


        /// <summary>
        /// Unloads the model.
        /// </summary>
        private async Task UnloadModel()
        {
            if (!_featureExtractorService.IsModelLoaded(SelectedModel.ModelSet))
                return;

            _logger.LogInformation($"'{SelectedModel.Name}' Unloading...");
            SelectedModel.IsLoading = true;
            await _featureExtractorService.UnloadModelAsync(SelectedModel.ModelSet);
            SelectedModel.IsLoading = false;
            SelectedModel.IsLoaded = false;
            _logger.LogInformation($"'{SelectedModel.Name}' Unloaded.");
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
