using Microsoft.Extensions.Logging;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
using OnnxStack.UI.Services;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;

namespace OnnxStack.UI.UserControls
{
    /// <summary>
    /// Interaction logic for Parameters.xaml
    /// </summary>
    public partial class ModelPickerControl : UserControl, INotifyPropertyChanged
    {
        private readonly ILogger<ModelPickerControl> _logger;
        private readonly IStableDiffusionService _stableDiffusionService;
        private ICollectionView _modelCollectionView;

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

        public OnnxStackUIConfig UISettings
        {
            get { return (OnnxStackUIConfig)GetValue(UISettingsProperty); }
            set { SetValue(UISettingsProperty, value); }
        }
        public static readonly DependencyProperty UISettingsProperty =
            DependencyProperty.Register("UISettings", typeof(OnnxStackUIConfig), typeof(ModelPickerControl), new PropertyMetadata((d, e) =>
            {
                if (d is ModelPickerControl control && e.NewValue is OnnxStackUIConfig settings)
                    control.InitializeModels();
            }));

        /// <summary>
        /// Gets or sets the supported diffusers.
        /// </summary>
        public List<DiffuserType> SupportedDiffusers
        {
            get { return (List<DiffuserType>)GetValue(SupportedDiffusersProperty); }
            set { SetValue(SupportedDiffusersProperty, value); }
        }
        public static readonly DependencyProperty SupportedDiffusersProperty =
            DependencyProperty.Register("SupportedDiffusers", typeof(List<DiffuserType>), typeof(ModelPickerControl), new PropertyMetadata((d, e) =>
            {
                if (d is ModelPickerControl control && e.NewValue is List<DiffuserType> diffusers)
                    control.ModelCollectionView?.Refresh();
            }));

        /// <summary>
        /// Gets or sets the selected model.
        /// </summary>
        public StableDiffusionModelSetViewModel SelectedModel
        {
            get { return (StableDiffusionModelSetViewModel)GetValue(SelectedModelProperty); }
            set { SetValue(SelectedModelProperty, value); }
        }
        public static readonly DependencyProperty SelectedModelProperty =
            DependencyProperty.Register("SelectedModel", typeof(StableDiffusionModelSetViewModel), typeof(ModelPickerControl));


        public ICollectionView ModelCollectionView
        {
            get { return _modelCollectionView; }
            set { _modelCollectionView = value; NotifyPropertyChanged(); }
        }


        private void InitializeModels()
        {
            ModelCollectionView = new ListCollectionView(UISettings.StableDiffusionModelSets);
            ModelCollectionView.Filter = (obj) =>
            {
                if (obj is not StableDiffusionModelSetViewModel viewModel)
                    return false;

                return viewModel.ModelSet.Diffusers.Intersect(SupportedDiffusers).Any();
            };
        }


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
                foreach (var model in UISettings.StableDiffusionModelSets.Where(x => x.IsLoaded))
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
