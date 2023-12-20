using Microsoft.Extensions.Logging;
using Microsoft.Win32;
using Models;
using OnnxStack.ImageUpscaler.Config;
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

        public MainWindow(OnnxStackUIConfig uiSettings, StableDiffusionConfig configuration, ImageUpscalerConfig upscaleConfiguration, ILogger<MainWindow> logger)
        {
            _logger = logger;
            UISettings = uiSettings;
            SaveImageCommand = new AsyncRelayCommand<UpscaleResult>(SaveUpscaleImageFile);
            SaveImageResultCommand = new AsyncRelayCommand<ImageResult>(SaveImageResultFile);
            SaveBlueprintCommand = new AsyncRelayCommand<ImageResult>(SaveBlueprintFile);
            NavigateTextToImageCommand = new AsyncRelayCommand<ImageResult>(NavigateTextToImage);
            NavigateImageToImageCommand = new AsyncRelayCommand<ImageResult>(NavigateImageToImage);
            NavigateImageInpaintCommand = new AsyncRelayCommand<ImageResult>(NavigateImageInpaint);
            NavigateImagePaintToImageCommand = new AsyncRelayCommand<ImageResult>(NavigateImagePaintToImage);
            NavigateUpscalerCommand = new AsyncRelayCommand<ImageResult>(NavigateUpscaler);
            WindowCloseCommand = new AsyncRelayCommand(WindowClose);
            WindowRestoreCommand = new AsyncRelayCommand(WindowRestore);
            WindowMinimizeCommand = new AsyncRelayCommand(WindowMinimize);
            WindowMaximizeCommand = new AsyncRelayCommand(WindowMaximize);
            InitializeComponent();
        }



        public AsyncRelayCommand WindowMinimizeCommand { get; }
        public AsyncRelayCommand WindowRestoreCommand { get; }
        public AsyncRelayCommand WindowMaximizeCommand { get; }
        public AsyncRelayCommand WindowCloseCommand { get; }
        public AsyncRelayCommand<UpscaleResult> SaveImageCommand { get; }
        public AsyncRelayCommand<ImageResult> SaveImageResultCommand { get; }
        public AsyncRelayCommand<ImageResult> SaveBlueprintCommand { get; }
        public AsyncRelayCommand<ImageResult> NavigateTextToImageCommand { get; }
        public AsyncRelayCommand<ImageResult> NavigateImageToImageCommand { get; }
        public AsyncRelayCommand<ImageResult> NavigateImageInpaintCommand { get; }
        public AsyncRelayCommand<ImageResult> NavigateImagePaintToImageCommand { get; }
        public AsyncRelayCommand<ImageResult> NavigateUpscalerCommand { get; }

        public OnnxStackUIConfig UISettings
        {
            get { return (OnnxStackUIConfig)GetValue(UISettingsProperty); }
            set { SetValue(UISettingsProperty, value); }
        }
        public static readonly DependencyProperty UISettingsProperty =
            DependencyProperty.Register("UISettings", typeof(OnnxStackUIConfig), typeof(MainWindow));


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
            await NavigateToTab(TabId.TextToImage, result);
        }

        private async Task NavigateImageToImage(ImageResult result)
        {
            await NavigateToTab(TabId.ImageToImage, result);
        }

        private async Task NavigateImageInpaint(ImageResult result)
        {
            await NavigateToTab(TabId.ImageInpaint, result);
        }

        private async Task NavigateImagePaintToImage(ImageResult result)
        {
            await NavigateToTab(TabId.PaintToImage, result);
        }

        private async Task NavigateUpscaler(ImageResult result)
        {
            await NavigateToTab(TabId.Upscaler, result);
        }

        private async Task NavigateToTab(TabId tab, ImageResult imageResult)
        {
            SelectedTabIndex = (int)tab;
            await SelectedTabItem.NavigateAsync(imageResult);
        }

        private enum TabId
        {
            TextToImage = 0,
            ImageToImage = 1,
            ImageInpaint = 2,
            PaintToImage = 3,
            Upscaler = 4
        }



        private async Task SaveImageResultFile(ImageResult imageResult)
        {
            try
            {
                var filename = GetSaveFilename($"image-{imageResult.SchedulerOptions.Seed}");
                if (string.IsNullOrEmpty(filename))
                    return;

                var result = await imageResult.Image.SaveImageFileAsync(filename);
                if (!result)
                    _logger.LogError("Error saving image");

            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error saving image");
            }
        }


        private async Task SaveUpscaleImageFile(UpscaleResult imageResult)
        {
            try
            {
                var filename = GetSaveFilename($"image-{imageResult.Info.OutputWidth}x{imageResult.Info.OutputHeight}");
                if (string.IsNullOrEmpty(filename))
                    return;

                var result = await imageResult.Image.SaveImageFileAsync(filename);
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
                {
                    _logger.LogInformation("Saving image blueprint canceled");
                    return;
                }

              

                var result = await imageResult.SaveBlueprintFileAsync(saveFileDialog.FileName);
                if (!result)
                    _logger.LogError("Error saving image blueprint");

            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error saving image blueprint");
            }
        }


        private string GetSaveFilename(string initialFilename)
        {
            var saveFileDialog = new SaveFileDialog
            {
                Title = Title,
                Filter = "png files (*.png)|*.png",
                DefaultExt = "png",
                AddExtension = true,
                RestoreDirectory = true,
                InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyPictures),
                FileName = $"{initialFilename}.png"
            };

            var dialogResult = saveFileDialog.ShowDialog();
            if (dialogResult == false)
            {
                _logger.LogInformation("Saving image canceled");
                return null;
            }

            return saveFileDialog.FileName;
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

        #region BaseWindow

        private Task WindowClose()
        {
            Close();
            return Task.CompletedTask;
        }

        private Task WindowRestore()
        {
            if (WindowState == WindowState.Maximized)
                WindowState = WindowState.Normal;
            else
                WindowState = WindowState.Maximized;
            return Task.CompletedTask;
        }

        private Task WindowMinimize()
        {
            WindowState = WindowState.Minimized;
            return Task.CompletedTask;
        }

        private Task WindowMaximize()
        {
            WindowState = WindowState.Maximized;
            return Task.CompletedTask;
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
