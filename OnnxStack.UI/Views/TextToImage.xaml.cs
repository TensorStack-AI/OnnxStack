using Microsoft.Extensions.Logging;
using Models;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace OnnxStack.UI.Views
{
    /// <summary>
    /// Interaction logic for TextToImageView.xaml
    /// </summary>
    public partial class TextToImageView : UserControl, INavigatable, INotifyPropertyChanged
    {
        private readonly ILogger<TextToImageView> _logger;
        private readonly IStableDiffusionService _stableDiffusionService;

        private bool _hasResult;
        private int _progressMax;
        private int _progressValue;
        private bool _isGenerating;
        private int _selectedTabIndex;
        private ImageResult _resultImage;
        private ModelOptionsModel _selectedModel;
        private PromptOptionsModel _promptOptionsModel;
        private SchedulerOptionsModel _schedulerOptions;
        private CancellationTokenSource _cancelationTokenSource;

        /// <summary>
        /// Initializes a new instance of the <see cref="TextToImageView"/> class.
        /// </summary>
        public TextToImageView()
        {
            if (!DesignerProperties.GetIsInDesignMode(this))
            {
                _logger = App.GetService<ILogger<TextToImageView>>();
                _stableDiffusionService = App.GetService<IStableDiffusionService>();
            }

            SupportedDiffusers = new() { DiffuserType.TextToImage };
            CancelCommand = new AsyncRelayCommand(Cancel, CanExecuteCancel);
            GenerateCommand = new AsyncRelayCommand(Generate, CanExecuteGenerate);
            ClearHistoryCommand = new AsyncRelayCommand(ClearHistory, CanExecuteClearHistory);
            PromptOptions = new PromptOptionsModel();
            SchedulerOptions = new SchedulerOptionsModel();
            ImageResults = new ObservableCollection<ImageResult>();
            ProgressMax = SchedulerOptions.InferenceSteps;
            InitializeComponent();
        }

        public OnnxStackUIConfig UISettings
        {
            get { return (OnnxStackUIConfig)GetValue(UISettingsProperty); }
            set { SetValue(UISettingsProperty, value); }
        }
        public static readonly DependencyProperty UISettingsProperty =
            DependencyProperty.Register("UISettings", typeof(OnnxStackUIConfig), typeof(TextToImageView));


        public List<DiffuserType> SupportedDiffusers { get; }
        public AsyncRelayCommand CancelCommand { get; }
        public AsyncRelayCommand GenerateCommand { get; }
        public AsyncRelayCommand ClearHistoryCommand { get; set; }
        public ObservableCollection<ImageResult> ImageResults { get; }

        public ModelOptionsModel SelectedModel
        {
            get { return _selectedModel; }
            set { _selectedModel = value; NotifyPropertyChanged(); }
        }

        public PromptOptionsModel PromptOptions
        {
            get { return _promptOptionsModel; }
            set { _promptOptionsModel = value; NotifyPropertyChanged(); }
        }

        public SchedulerOptionsModel SchedulerOptions
        {
            get { return _schedulerOptions; }
            set { _schedulerOptions = value; NotifyPropertyChanged(); }
        }

        public ImageResult ResultImage
        {
            get { return _resultImage; }
            set { _resultImage = value; NotifyPropertyChanged(); }
        }

        public int ProgressValue
        {
            get { return _progressValue; }
            set { _progressValue = value; NotifyPropertyChanged(); }
        }

        public int ProgressMax
        {
            get { return _progressMax; }
            set { _progressMax = value; NotifyPropertyChanged(); }
        }

        public bool IsGenerating
        {
            get { return _isGenerating; }
            set { _isGenerating = value; NotifyPropertyChanged(); }
        }

        public bool HasResult
        {
            get { return _hasResult; }
            set { _hasResult = value; NotifyPropertyChanged(); }
        }

        public int SelectedTabIndex
        {
            get { return _selectedTabIndex; }
            set { _selectedTabIndex = value; NotifyPropertyChanged(); }
        }

     


        /// <summary>
        /// Called on Navigate
        /// </summary>
        /// <param name="imageResult">The image result.</param>
        /// <returns></returns>
        public Task NavigateAsync(ImageResult imageResult)
        {
            Reset();
            HasResult = false;
            ResultImage = null;
            if (imageResult.Model.ModelOptions.Diffusers.Contains(DiffuserType.TextToImage))
            {
                SelectedModel = imageResult.Model;
            }
            PromptOptions = new PromptOptionsModel
            {
                Prompt = imageResult.Prompt,
                NegativePrompt = imageResult.NegativePrompt
            };
            SchedulerOptions = imageResult.SchedulerOptions.ToSchedulerOptionsModel();
            SelectedTabIndex = 0;
            return Task.CompletedTask;
        }


        /// <summary>
        /// Generates this image result.
        /// </summary>
        private async Task Generate()
        {
            HasResult = false;
            IsGenerating = true;
            ResultImage = null;
            var promptOptions = new PromptOptions
            {
                Prompt = PromptOptions.Prompt,
                NegativePrompt = PromptOptions.NegativePrompt,
                DiffuserType = DiffuserType.TextToImage
            };

            var schedulerOptions = SchedulerOptions.ToSchedulerOptions();
            var resultImage = await ExecuteStableDiffusion(_selectedModel.ModelOptions, promptOptions, schedulerOptions);
            if (resultImage != null)
            {
                ResultImage = resultImage;
                ImageResults.Add(resultImage);
                HasResult = true;
            }

            Reset();
        }

        /// <summary>
        /// Determines whether this instance can execute Generate.
        /// </summary>
        /// <returns>
        ///   <c>true</c> if this instance can execute Generate; otherwise, <c>false</c>.
        /// </returns>
        private bool CanExecuteGenerate()
        {
            return !IsGenerating && !string.IsNullOrEmpty(PromptOptions.Prompt);
        }


        /// <summary>
        /// Cancels this generation.
        /// </summary>
        /// <returns></returns>
        private Task Cancel()
        {
            _cancelationTokenSource?.Cancel();
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can execute Cancel.
        /// </summary>
        /// <returns>
        ///   <c>true</c> if this instance can execute Cancel; otherwise, <c>false</c>.
        /// </returns>
        private bool CanExecuteCancel()
        {
            return IsGenerating;
        }


        /// <summary>
        /// Clears the history.
        /// </summary>
        /// <returns></returns>
        private Task ClearHistory()
        {
            ImageResults.Clear();
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can execute ClearHistory.
        /// </summary>
        /// <returns>
        ///   <c>true</c> if this instance can execute ClearHistory; otherwise, <c>false</c>.
        /// </returns>
        private bool CanExecuteClearHistory()
        {
            return ImageResults.Count > 0;
        }


        /// <summary>
        /// Resets this instance.
        /// </summary>
        private void Reset()
        {
            IsGenerating = false;
            ProgressValue = 0;
        }


        /// <summary>
        /// Executes the stable diffusion.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <returns></returns>
        private async Task<ImageResult> ExecuteStableDiffusion(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions)
        {
            try
            {
                var timestamp = Stopwatch.GetTimestamp();
                _cancelationTokenSource = new CancellationTokenSource();
                var result = await _stableDiffusionService.GenerateAsBytesAsync(modelOptions, promptOptions, schedulerOptions, ProgressCallback(), _cancelationTokenSource.Token);
                if (result == null)
                    return null;

                var image = Utils.CreateBitmap(result);
                if (image == null)
                    return null;

                var imageResult = new ImageResult
                {
                    Image = image,
                    Model = _selectedModel,
                    Prompt = promptOptions.Prompt,
                    NegativePrompt = promptOptions.NegativePrompt,
                    PipelineType = _selectedModel.ModelOptions.PipelineType,
                    DiffuserType = promptOptions.DiffuserType,
                    SchedulerType = schedulerOptions.SchedulerType,
                    SchedulerOptions = schedulerOptions,
                    Elapsed = Stopwatch.GetElapsedTime(timestamp).TotalSeconds
                };

                if (UISettings.ImageAutoSave)
                    await imageResult.AutoSave(Path.Combine(UISettings.ImageAutoSaveDirectory, "TextToImage"), UISettings.ImageAutoSaveBlueprint);

                return imageResult;

            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating image");
                return null;
            }
        }


        /// <summary>
        /// StableDiffusion progress callback.
        /// </summary>
        /// <returns></returns>
        private Action<int, int> ProgressCallback()
        {
            return (value, maximum) =>
            {
                App.UIInvoke(() =>
                {
                    if (_cancelationTokenSource.IsCancellationRequested)
                        return;

                    if (ProgressMax != maximum)
                        ProgressMax = maximum;

                    ProgressValue = value;
                });
            };
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