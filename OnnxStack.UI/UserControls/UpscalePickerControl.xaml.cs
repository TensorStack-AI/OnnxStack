using Microsoft.Extensions.Logging;
using OnnxStack.Core;
using OnnxStack.ImageUpscaler.Services;
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
    /// Interaction logic for UpscalePickerControl.xaml
    /// </summary>
    public partial class UpscalePickerControl : UserControl, INotifyPropertyChanged
    {
        private readonly ILogger<UpscalePickerControl> _logger;
        private readonly IUpscaleService _upscaleService;

        /// <summary>Initializes a new instance of the <see cref="UpscalePickerControl" /> class.</summary>
        public UpscalePickerControl()
        {
            if (!DesignerProperties.GetIsInDesignMode(this))
            {
                _logger = App.GetService<ILogger<UpscalePickerControl>>();
                _upscaleService = App.GetService<IUpscaleService>();
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
            DependencyProperty.Register("UISettings", typeof(OnnxStackUIConfig), typeof(UpscalePickerControl));


        /// <summary>
        /// Gets or sets the selected model.
        /// </summary>
        public UpscaleModelSetViewModel SelectedModel
        {
            get { return (UpscaleModelSetViewModel)GetValue(SelectedModelProperty); }
            set { SetValue(SelectedModelProperty, value); }
        }
        public static readonly DependencyProperty SelectedModelProperty =
            DependencyProperty.Register("SelectedModel", typeof(UpscaleModelSetViewModel), typeof(UpscalePickerControl));



        /// <summary>
        /// Loads the model.
        /// </summary>
        private async Task LoadModel()
        {
            if (_upscaleService.IsModelLoaded(SelectedModel.ModelSet))
                return;

            var elapsed = _logger.LogBegin($"'{SelectedModel.Name}' Loading...");
            SelectedModel.IsLoaded = false;
            SelectedModel.IsLoading = true;

            try
            {
                if (UISettings.ModelCacheMode == ModelCacheMode.Single)
                {
                    foreach (var model in UISettings.UpscaleModelSets.Where(x => x.IsLoaded))
                    {
                        _logger.LogInformation($"'{model.Name}' Unloading...");
                        await _upscaleService.UnloadModelAsync(model.ModelSet);
                        model.IsLoaded = false;
                    }
                }

                SelectedModel.ModelSet.ApplyConfigurationOverrides();
                await _upscaleService.AddModelAsync(SelectedModel.ModelSet);
                SelectedModel.IsLoaded = await _upscaleService.LoadModelAsync(SelectedModel.ModelSet);
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
            if (!_upscaleService.IsModelLoaded(SelectedModel.ModelSet))
                return;

            _logger.LogInformation($"'{SelectedModel.Name}' Unloading...");
            SelectedModel.IsLoading = true;
            await _upscaleService.UnloadModelAsync(SelectedModel.ModelSet);
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
