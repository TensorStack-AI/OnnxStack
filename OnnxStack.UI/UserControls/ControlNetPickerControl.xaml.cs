using Microsoft.Extensions.Logging;
using OnnxStack.Core;
using OnnxStack.ImageUpscaler.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
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
    /// Interaction logic for ControlNetPickerControl.xaml
    /// </summary>
    public partial class ControlNetPickerControl : UserControl, INotifyPropertyChanged
    {
        private readonly ILogger<ControlNetPickerControl> _logger;
        private readonly IStableDiffusionService _stableDiffusionService;

        /// <summary>Initializes a new instance of the <see cref="ControlNetPickerControl" /> class.</summary>
        public ControlNetPickerControl()
        {
            if (!DesignerProperties.GetIsInDesignMode(this))
            {
                _logger = App.GetService<ILogger<ControlNetPickerControl>>();
                _stableDiffusionService = App.GetService<IStableDiffusionService>();
            }

            LoadCommand = new AsyncRelayCommand(LoadModel);
            UnloadCommand = new AsyncRelayCommand(UnloadModel);
            InitializeComponent();
        }

        public AsyncRelayCommand LoadCommand { get; set; }
        public AsyncRelayCommand UnloadCommand { get; set; }

        public OnnxStackUIConfig UISettings
        {
            get { return (OnnxStackUIConfig)GetValue(UISettingsProperty); }
            set { SetValue(UISettingsProperty, value); }
        }
        public static readonly DependencyProperty UISettingsProperty =
            DependencyProperty.Register("UISettings", typeof(OnnxStackUIConfig), typeof(ControlNetPickerControl));


        /// <summary>
        /// Gets or sets the selected model.
        /// </summary>
        public ControlNetModelSetViewModel SelectedModel
        {
            get { return (ControlNetModelSetViewModel)GetValue(SelectedModelProperty); }
            set { SetValue(SelectedModelProperty, value); }
        }
        public static readonly DependencyProperty SelectedModelProperty =
            DependencyProperty.Register("SelectedModel", typeof(ControlNetModelSetViewModel), typeof(ControlNetPickerControl));



        /// <summary>
        /// Loads the model.
        /// </summary>
        private async Task LoadModel()
        {
            if (_stableDiffusionService.IsModelLoaded(SelectedModel.ModelSet))
                return;

            var elapsed = _logger.LogBegin($"'{SelectedModel.Name}' Loading...");
            SelectedModel.IsLoaded = false;
            SelectedModel.IsLoading = true;

            try
            {
                foreach (var model in UISettings.ControlNetModelSets.Where(x => x.IsLoaded))
                {
                    _logger.LogInformation($"'{model.Name}' Unloading...");
                    await _stableDiffusionService.UnloadModelAsync(model.ModelSet);
                    model.IsLoaded = false;
                }
                SelectedModel.IsLoaded = await _stableDiffusionService.LoadModelAsync(SelectedModel.ModelSet);
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
            if (!_stableDiffusionService.IsModelLoaded(SelectedModel.ModelSet))
                return;

            _logger.LogInformation($"'{SelectedModel.Name}' Unloading...");
            SelectedModel.IsLoading = true;
            await _stableDiffusionService.UnloadModelAsync(SelectedModel.ModelSet);
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
