using Microsoft.Extensions.Logging;
using OnnxStack.Core.Config;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Dialogs;
using OnnxStack.UI.Models;
using OnnxStack.UI.Services;
using System;
using System.ComponentModel;
using System.Linq;
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
        private readonly IDialogService _dialogService;
        private readonly IDeviceService _deviceService;
        private StableDiffusionModelSetViewModel _selectedStableDiffusionModel;
        private UpscaleModelSetViewModel _selectedUpscaleModel;
        private ControlNetModelSetViewModel _selectedControlNetModel;

        /// <summary>
        /// Initializes a new instance of the <see cref="SettingsView"/> class.
        /// </summary>
        public SettingsView()
        {
            if (!DesignerProperties.GetIsInDesignMode(this))
            {
                _logger = App.GetService<ILogger<SettingsView>>();
                _dialogService = App.GetService<IDialogService>();
                _deviceService = App.GetService<IDeviceService>();
             
            }

            SaveCommand = new AsyncRelayCommand(SaveConfigurationFile);
            AddUpscaleModelCommand = new AsyncRelayCommand(AddUpscaleModel);
            UpdateUpscaleModelCommand = new AsyncRelayCommand(UpdateUpscaleModel, () => SelectedUpscaleModel is not null);
            RemoveUpscaleModelCommand = new AsyncRelayCommand(RemoveUpscaleModel, () => SelectedUpscaleModel is not null);
            AddStableDiffusionModelCommand = new AsyncRelayCommand(AddStableDiffusionModel);
            UpdateStableDiffusionModelCommand = new AsyncRelayCommand(UpdateStableDiffusionModel, () => SelectedStableDiffusionModel is not null);
            RemoveStableDiffusionModelCommand = new AsyncRelayCommand(RemoveStableDiffusionModel, () => SelectedStableDiffusionModel is not null);

            AddControlNetModelCommand = new AsyncRelayCommand(AddControlNetModel);
            UpdateControlNetModelCommand = new AsyncRelayCommand(UpdateControlNetModel, () => SelectedControlNetModel is not null);
            RemoveControlNetModelCommand = new AsyncRelayCommand(RemoveControlNetModel, () => SelectedControlNetModel is not null);
            InitializeComponent();
        }

        public AsyncRelayCommand SaveCommand { get; }
        public AsyncRelayCommand AddUpscaleModelCommand { get; }
        public AsyncRelayCommand UpdateUpscaleModelCommand { get; }
        public AsyncRelayCommand RemoveUpscaleModelCommand { get; }
        public AsyncRelayCommand AddStableDiffusionModelCommand { get; }
        public AsyncRelayCommand UpdateStableDiffusionModelCommand { get; }
        public AsyncRelayCommand RemoveStableDiffusionModelCommand { get; }
        public AsyncRelayCommand AddControlNetModelCommand { get; }
        public AsyncRelayCommand UpdateControlNetModelCommand { get; }
        public AsyncRelayCommand RemoveControlNetModelCommand { get; }

        public OnnxStackUIConfig UISettings
        {
            get { return (OnnxStackUIConfig)GetValue(UISettingsProperty); }
            set { SetValue(UISettingsProperty, value); }
        }
        public static readonly DependencyProperty UISettingsProperty =
            DependencyProperty.Register("UISettings", typeof(OnnxStackUIConfig), typeof(SettingsView));


        public StableDiffusionModelSetViewModel SelectedStableDiffusionModel
        {
            get { return _selectedStableDiffusionModel; }
            set { _selectedStableDiffusionModel = value; NotifyPropertyChanged(); }
        }
             
        public UpscaleModelSetViewModel SelectedUpscaleModel
        {
            get { return _selectedUpscaleModel; }
            set { _selectedUpscaleModel = value; NotifyPropertyChanged(); }
        }

        public ControlNetModelSetViewModel SelectedControlNetModel
        {
            get { return _selectedControlNetModel; }
            set { _selectedControlNetModel = value; NotifyPropertyChanged(); }
        }

        public Task NavigateAsync(ImageResult imageResult)
        {
            throw new NotImplementedException();
        }

        private Task SaveConfigurationFile()
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


        #region StableDiffusion

        private async Task AddStableDiffusionModel()
        {
            var addModelDialog = _dialogService.GetDialog<AddModelDialog>();
            if (addModelDialog.ShowDialog())
            {
                var model = new StableDiffusionModelSetViewModel
                {
                    Name = addModelDialog.ModelSetResult.Name,
                    ModelSet = addModelDialog.ModelSetResult
                };

                UISettings.StableDiffusionModelSets.Add(model);
                SelectedStableDiffusionModel = model;
                await SaveConfigurationFile();
            }
        }

        private async Task RemoveStableDiffusionModel()
        {
            if (SelectedStableDiffusionModel.IsLoaded)
            {
                MessageBox.Show("Please unload model before uninstalling", "Model In Use", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            UISettings.StableDiffusionModelSets.Remove(SelectedStableDiffusionModel);
            SelectedStableDiffusionModel = UISettings.StableDiffusionModelSets.FirstOrDefault();
            await SaveConfigurationFile();
        }


        private async Task UpdateStableDiffusionModel()
        {
            if (SelectedStableDiffusionModel.IsLoaded)
            {
                MessageBox.Show("Please unload model before updating", "Model In Use", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            var updateModelDialog = _dialogService.GetDialog<UpdateModelDialog>();
            if (updateModelDialog.ShowDialog(SelectedStableDiffusionModel.ModelSet))
            {
                var modelSet = updateModelDialog.ModelSetResult;
                SelectedStableDiffusionModel.ModelSet = modelSet;
                SelectedStableDiffusionModel.Name = modelSet.Name;
                UISettings.StableDiffusionModelSets.ForceNotifyCollectionChanged();
                await SaveConfigurationFile();
            }
        }

        #endregion


        #region Upscale

        private async Task AddUpscaleModel()
        {
            var addModelDialog = _dialogService.GetDialog<AddUpscaleModelDialog>();
            if (addModelDialog.ShowDialog())
            {
                var model = new UpscaleModelSetViewModel
                {
                    Name = addModelDialog.ModelSetResult.Name,
                    ModelSet = addModelDialog.ModelSetResult
                };
                UISettings.UpscaleModelSets.Add(model);
                SelectedUpscaleModel = model;
                await SaveConfigurationFile();
            }
        }



        private async Task UpdateUpscaleModel()
        {
            if (SelectedUpscaleModel.IsLoaded)
            {
                MessageBox.Show("Please unload model before updating", "Model In Use", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            var updateModelDialog = _dialogService.GetDialog<UpdateUpscaleModelDialog>();
            if (updateModelDialog.ShowDialog(SelectedUpscaleModel.ModelSet))
            {
                var modelSet = updateModelDialog.ModelSetResult;
                SelectedUpscaleModel.ModelSet = modelSet;
                SelectedUpscaleModel.Name = modelSet.Name;
                UISettings.UpscaleModelSets.ForceNotifyCollectionChanged();
                await SaveConfigurationFile();
            }
        }


        private async Task RemoveUpscaleModel()
        {
            if (SelectedUpscaleModel.IsLoaded)
            {
                MessageBox.Show("Please unload model before uninstalling", "Model In Use", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            UISettings.UpscaleModelSets.Remove(SelectedUpscaleModel);
            SelectedUpscaleModel = UISettings.UpscaleModelSets.FirstOrDefault();
            await SaveConfigurationFile();
        }

        #endregion


        #region ControlNet

        private async Task AddControlNetModel()
        {
            var addModelDialog = _dialogService.GetDialog<AddControlNetModelDialog>();
            if (addModelDialog.ShowDialog())
            {
                var model = new ControlNetModelSetViewModel
                {
                    Name = addModelDialog.ModelSetResult.Name,
                    ModelSet = addModelDialog.ModelSetResult
                };
                UISettings.ControlNetModelSets.Add(model);
                SelectedControlNetModel = model;
                await SaveConfigurationFile();
            }
        }


        private async Task UpdateControlNetModel()
        {
            if (SelectedControlNetModel.IsLoaded)
            {
                MessageBox.Show("Please unload model before updating", "Model In Use", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            var updateModelDialog = _dialogService.GetDialog<UpdateControlNetModelDialog>();
            if (updateModelDialog.ShowDialog(SelectedControlNetModel.ModelSet))
            {
                var modelSet = updateModelDialog.ModelSetResult;
                SelectedControlNetModel.ModelSet = modelSet;
                SelectedControlNetModel.Name = modelSet.Name;
                UISettings.ControlNetModelSets.ForceNotifyCollectionChanged();
                await SaveConfigurationFile();
            }
        }


        private async Task RemoveControlNetModel()
        {
            if (SelectedControlNetModel.IsLoaded)
            {
                MessageBox.Show("Please unload model before uninstalling", "Model In Use", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            UISettings.ControlNetModelSets.Remove(SelectedControlNetModel);
            SelectedControlNetModel = UISettings.ControlNetModelSets.FirstOrDefault();
            await SaveConfigurationFile();
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
