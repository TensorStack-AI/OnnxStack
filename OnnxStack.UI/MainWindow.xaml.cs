using Microsoft.Extensions.Logging;
using Microsoft.Win32;
using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
using OnnxStack.UI.Views;
using System;
using System.ComponentModel;
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
        private readonly OnnxStackConfig _configuration;

        public MainWindow(OnnxStackConfig configuration, ILogger<MainWindow> logger)
        {
            _logger = logger;
            _configuration = configuration;
            SaveImageCommand = new AsyncRelayCommand<ImageResult>(SaveImageFile);
            SaveBlueprintCommand = new AsyncRelayCommand<ImageResult>(SaveBlueprintFile);
            NavigateTextToImageCommand = new AsyncRelayCommand<ImageResult>(NavigateTextToImage);
            NavigateImageToImageCommand = new AsyncRelayCommand<ImageResult>(NavigateImageToImage);
            NavigateImageInpaintCommand = new AsyncRelayCommand<ImageResult>(NavigateImageInpaint);
            NavigateImageUpscaleCommand = new AsyncRelayCommand<ImageResult>(NavigateImageUpscale);
            InitializeComponent();
        }



        public AsyncRelayCommand<ImageResult> SaveImageCommand { get; }
        public AsyncRelayCommand<ImageResult> SaveBlueprintCommand { get; }
        public AsyncRelayCommand<ImageResult> NavigateTextToImageCommand { get; }
        public AsyncRelayCommand<ImageResult> NavigateImageToImageCommand { get; }
        public AsyncRelayCommand<ImageResult> NavigateImageInpaintCommand { get; }
        public AsyncRelayCommand<ImageResult> NavigateImageUpscaleCommand { get; }

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
            await NavigateToTab(ProcessType.TextToImage, result);
        }

        private async Task NavigateImageToImage(ImageResult result)
        {
            await NavigateToTab(ProcessType.ImageToImage, result);
        }


        private async Task NavigateImageInpaint(ImageResult result)
        {
            await NavigateToTab(ProcessType.ImageInpaint, result);
        }


        private Task NavigateImageUpscale(ImageResult result)
        {
            return Task.CompletedTask;
        }


        private async Task NavigateToTab(ProcessType processType, ImageResult imageResult)
        {
            SelectedTabIndex = (int)processType;
            await SelectedTabItem.NavigateAsync(imageResult);
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
