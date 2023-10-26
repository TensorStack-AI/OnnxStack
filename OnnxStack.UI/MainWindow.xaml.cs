using Microsoft.Extensions.Logging;
using Microsoft.Win32;
using Models;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
using OnnxStack.UI.Views;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;

namespace OnnxStack.UI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window, INotifyPropertyChanged
    {
        private string _outputLog;
        private int _selectedTabIndex;
        private INavigatable _selectedTabItem;
        private readonly ILogger<MainWindow> _logger;
        private ObservableCollection<ModelOptionsModel> _models;

        public MainWindow(StableDiffusionConfig configuration, ILogger<MainWindow> logger)
        {
            _logger = logger;
            SaveImageCommand = new AsyncRelayCommand<ImageResult>(SaveImageFile);
            SaveBlueprintCommand = new AsyncRelayCommand<ImageResult>(SaveBlueprintFile);
            NavigateTextToImageCommand = new AsyncRelayCommand<ImageResult>(NavigateTextToImage);
            NavigateImageToImageCommand = new AsyncRelayCommand<ImageResult>(NavigateImageToImage);
            NavigateImageInpaintCommand = new AsyncRelayCommand<ImageResult>(NavigateImageInpaint);
            NavigateImageUpscaleCommand = new AsyncRelayCommand<ImageResult>(NavigateImageUpscale);
            Models = CreateModelOptions(configuration.OnnxModelSets);
            InitializeComponent();
        }

        public AsyncRelayCommand<ImageResult> SaveImageCommand { get; }
        public AsyncRelayCommand<ImageResult> SaveBlueprintCommand { get; }
        public AsyncRelayCommand<ImageResult> NavigateTextToImageCommand { get; }
        public AsyncRelayCommand<ImageResult> NavigateImageToImageCommand { get; }
        public AsyncRelayCommand<ImageResult> NavigateImageInpaintCommand { get; }
        public AsyncRelayCommand<ImageResult> NavigateImageUpscaleCommand { get; }

        public ObservableCollection<ModelOptionsModel> Models
        {
            get { return _models; }
            set { _models = value; NotifyPropertyChanged(); }
        }

        public int SelectedTabIndex
        {
            get { return _selectedTabIndex; }
            set { _selectedTabIndex = value; NotifyPropertyChanged(); }
        }

        public INavigatable SelectedTabItem
        {
            get { return _selectedTabItem; }
            set { _selectedTabItem = value; NotifyPropertyChanged(); }
        }

        private async Task NavigateTextToImage(ImageResult result)
        {
            await NavigateToTab(DiffuserType.TextToImage, result);
        }

        private async Task NavigateImageToImage(ImageResult result)
        {
            await NavigateToTab(DiffuserType.ImageToImage, result);
        }

        private async Task NavigateImageInpaint(ImageResult result)
        {
            await NavigateToTab(DiffuserType.ImageInpaint, result);
        }

        private Task NavigateImageUpscale(ImageResult result)
        {
            return Task.CompletedTask;
        }


        private async Task NavigateToTab(DiffuserType diffuserType, ImageResult imageResult)
        {
            SelectedTabIndex = (int)diffuserType;
            await SelectedTabItem.NavigateAsync(imageResult);
        }

        private ObservableCollection<ModelOptionsModel> CreateModelOptions(List<ModelOptions> onnxModelSets)
        {
            var models = onnxModelSets
            .Select(model => new ModelOptionsModel
            {
                Name = model.Name,
                ModelOptions = model,
                IsEnabled = model.IsEnabled
            });
            return new ObservableCollection<ModelOptionsModel>(models);
        }

        private async Task SaveImageFile(ImageResult imageResult)
        {
            try
            {
                var saveFileDialog = new SaveFileDialog
                {
                    Title = Title,
                    Filter = "png files (*.png)|*.png",
                    DefaultExt = "png",
                    AddExtension = true,
                    RestoreDirectory = true,
                    InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyPictures),
                    FileName = $"image-{imageResult.SchedulerOptions.Seed}.png"
                };

                var dialogResult = saveFileDialog.ShowDialog();
                if (dialogResult == false)
                    _logger.LogInformation("Saving image canceled");

                var result = await imageResult.SaveImageFile(saveFileDialog.FileName);
                if (!result)
                    _logger.LogError("Error saving image");

            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error saving image");
            }
        }


        private async Task SaveBlueprintFile(ImageResult imageResult)
        {
            try
            {
                var saveFileDialog = new SaveFileDialog
                {
                    Title = Title,
                    Filter = "json files (*.json)|*.json",
                    DefaultExt = "json",
                    AddExtension = true,
                    RestoreDirectory = true,
                    InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyPictures),
                    FileName = $"image-{imageResult.SchedulerOptions.Seed}.json"
                };

                var dialogResult = saveFileDialog.ShowDialog();
                if (dialogResult == false)
                    _logger.LogInformation("Saving image blueprint canceled");

                var result = await imageResult.SaveBlueprintFile(saveFileDialog.FileName);
                if (!result)
                    _logger.LogError("Error saving image blueprint");

            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error saving image blueprint");
            }
        }


        /// <summary>
        /// Gets or sets the output log.
        /// </summary>
        public string OutputLog
        {
            get { return _outputLog; }
            set { _outputLog = value; NotifyPropertyChanged(); }
        }


        /// <summary>
        /// Updates the output log.
        /// </summary>
        /// <param name="message">The message.</param>
        public void UpdateOutputLog(string message)
        {
            OutputLog += message;
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
