using Microsoft.Extensions.Logging;
using Models;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace OnnxStack.UI.UserControls
{
    /// <summary>
    /// Interaction logic for Parameters.xaml
    /// </summary>
    public partial class ModelPickerControl : UserControl, INotifyPropertyChanged
    {
        private readonly ILogger<ModelPickerControl> _logger;
        private readonly IStableDiffusionService _stableDiffusionService;

        /// <summary>Initializes a new instance of the <see cref="ModelPickerControl" /> class.</summary>
        public ModelPickerControl()
        {
            if (!DesignerProperties.GetIsInDesignMode(this))
            {
                _logger = App.GetService<ILogger<ModelPickerControl>>();
                _stableDiffusionService = App.GetService<IStableDiffusionService>();
            }

            LoadCommand = new AsyncRelayCommand(LoadModel);
            UnloadCommand = new AsyncRelayCommand(UnloadModel);
            InitializeComponent();
        }

        public AsyncRelayCommand LoadCommand { get; set; }
        public AsyncRelayCommand UnloadCommand { get; set; }


        /// <summary>
        /// Gets or sets the models.
        /// </summary>
        public ObservableCollection<ModelOptionsModel> Models
        {
            get { return (ObservableCollection<ModelOptionsModel>)GetValue(ModelsProperty); }
            set { SetValue(ModelsProperty, value); }
        }
        public static readonly DependencyProperty ModelsProperty =
            DependencyProperty.Register("Models", typeof(ObservableCollection<ModelOptionsModel>), typeof(ModelPickerControl), new PropertyMetadata(propertyChangedCallback: OnModelsChanged));


        public OnnxStackUIConfig UISettings
        {
            get { return (OnnxStackUIConfig)GetValue(UISettingsProperty); }
            set { SetValue(UISettingsProperty, value); }
        }
        public static readonly DependencyProperty UISettingsProperty =
            DependencyProperty.Register("UISettings", typeof(OnnxStackUIConfig), typeof(ModelPickerControl));



        /// <summary>
        /// Gets or sets the supported diffusers.
        /// </summary>
        public List<DiffuserType> SupportedDiffusers
        {
            get { return (List<DiffuserType>)GetValue(SupportedDiffusersProperty); }
            set { SetValue(SupportedDiffusersProperty, value); }
        }
        public static readonly DependencyProperty SupportedDiffusersProperty =
            DependencyProperty.Register("SupportedDiffusers", typeof(List<DiffuserType>), typeof(ModelPickerControl));


        /// <summary>
        /// Gets or sets the selected model.
        /// </summary>
        public ModelOptionsModel SelectedModel
        {
            get { return (ModelOptionsModel)GetValue(SelectedModelProperty); }
            set { SetValue(SelectedModelProperty, value); }
        }
        public static readonly DependencyProperty SelectedModelProperty =
            DependencyProperty.Register("SelectedModel", typeof(ModelOptionsModel), typeof(ModelPickerControl));



        /// <summary>
        /// Loads the model.
        /// </summary>
        private async Task LoadModel()
        {
            if (_stableDiffusionService.IsModelLoaded(SelectedModel.ModelOptions))
                return;

            var elapsed = _logger.LogBegin($"'{SelectedModel.Name}' Loading...");
            SelectedModel.IsLoaded = false;
            SelectedModel.IsLoading = true;

            try
            {
                if (UISettings.ModelCacheMode == ModelCacheMode.Single)
                {
                    foreach (var model in Models.Where(x => x.IsLoaded))
                    {
                        _logger.LogInformation($"'{model.Name}' Unloading...");
                        await _stableDiffusionService.UnloadModel(model.ModelOptions);
                    }
                }

                SelectedModel.IsLoaded = await _stableDiffusionService.LoadModel(SelectedModel.ModelOptions);
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
            if (!_stableDiffusionService.IsModelLoaded(SelectedModel.ModelOptions))
                return;

            _logger.LogInformation($"'{SelectedModel.Name}' Unloading...");
            SelectedModel.IsLoading = true;
            await _stableDiffusionService.UnloadModel(SelectedModel.ModelOptions);
            SelectedModel.IsLoading = false;
            SelectedModel.IsLoaded = false;
            _logger.LogInformation($"'{SelectedModel.Name}' Unloaded.");
        }


        /// <summary>
        /// Called when the Models source collection has changes, via Settings most likely.
        /// </summary>
        /// <param name="owner">The owner.</param>
        /// <param name="e">The <see cref="DependencyPropertyChangedEventArgs"/> instance containing the event data.</param>
        private static void OnModelsChanged(DependencyObject owner, DependencyPropertyChangedEventArgs e)
        {
            if (owner is ModelPickerControl control)
            {
                control.SelectedModel = control.Models
                    .Where(x => control.SupportedDiffusers.Any(x.ModelOptions.Diffusers.Contains))
                    .FirstOrDefault(x => x.ModelOptions.IsEnabled);
            }
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
