using Microsoft.Extensions.Logging;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
using OnnxStack.UI.Services;
using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Controls;

namespace OnnxStack.UI.Views
{
    /// <summary>
    /// Interaction logic for TextToImageView.xaml
    /// </summary>
    public partial class TextToImageView : UserControl, INavigatable, INotifyPropertyChanged
    {
        private readonly ILogger<TextToImageView> _logger;
        private readonly IDialogService _dialogService;
        private readonly IStableDiffusionService _stableDiffusionService;

        private bool _hasResult;
        private int _progressMax;
        private int _progressValue;
        private bool _isGenerating;
        private int _selectedTabIndex;
        private ImageResult _resultImage;
        private PromptOptionsModel _promptOptionsModel;
        private SchedulerOptionsModel _schedulerOptions;
        private CancellationTokenSource _cancelationTokenSource;

        public TextToImageView()
        {
            _logger = App.GetService<ILogger<TextToImageView>>();
            _dialogService = App.GetService<IDialogService>();
            _stableDiffusionService = App.GetService<IStableDiffusionService>();
            CancelCommand = new AsyncRelayCommand(Cancel, CanExecuteCancel);
            GenerateCommand = new AsyncRelayCommand(Generate, CanExecuteGenerate);
            ClearHistoryCommand = new AsyncRelayCommand(ClearHistory, CanExecuteClearHistory);
            PromptOptions = new PromptOptionsModel();
            SchedulerOptions = new SchedulerOptionsModel();
            ImageResults = new ObservableCollection<ImageResult>();
            ProgressMax = SchedulerOptions.InferenceSteps;
            InitializeComponent();
        }

        public AsyncRelayCommand CancelCommand { get; }
        public AsyncRelayCommand GenerateCommand { get; }
        public AsyncRelayCommand ClearHistoryCommand { get; set; }
        public ObservableCollection<ImageResult> ImageResults { get; }

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


        public Task NavigateAsync(ImageResult imageResult)
        {
            Reset();
            HasResult = false;
            ResultImage = null;
            PromptOptions = new PromptOptionsModel
            {
                Prompt = imageResult.Prompt,
                NegativePrompt = imageResult.NegativePrompt,
                SchedulerType = imageResult.SchedulerType
            };
            SchedulerOptions = imageResult.SchedulerOptions.ToSchedulerOptionsModel();
            SelectedTabIndex = 0;
            return Task.CompletedTask;
        }


        private async Task Generate()
        {
            HasResult = false;
            IsGenerating = true;
            ResultImage = null;
            var promptOptions = new PromptOptions
            {
                Prompt = PromptOptions.Prompt,
                NegativePrompt = PromptOptions.NegativePrompt,
                SchedulerType = PromptOptions.SchedulerType,
                ProcessType = ProcessType.TextToImage
            };

            var schedulerOptions = SchedulerOptions.ToSchedulerOptions();
            var resultImage = await GenerateResult(promptOptions, schedulerOptions);
            if (resultImage != null)
            {
                ResultImage = resultImage;
                ImageResults.Add(resultImage);
                HasResult = true;
            }

            App.UIInvoke(Reset);
        }


        private bool CanExecuteGenerate()
        {
            return !IsGenerating && !string.IsNullOrEmpty(PromptOptions.Prompt);
        }


        private Task Cancel()
        {
            _cancelationTokenSource?.Cancel();
            return Task.CompletedTask;
        }


        private bool CanExecuteCancel()
        {
            return IsGenerating;
        }

        private Task ClearHistory()
        {
            ImageResults.Clear();
            return Task.CompletedTask;
        }


        private bool CanExecuteClearHistory()
        {
            return ImageResults.Count > 0;
        }


        private void Reset()
        {
            IsGenerating = false;
            ProgressValue = 0;
        }

        private async Task<ImageResult> GenerateResult(PromptOptions promptOptions, SchedulerOptions schedulerOptions)
        {
            try
            {
                var timestamp = Stopwatch.GetTimestamp();
                _cancelationTokenSource = new CancellationTokenSource();
                var result = await _stableDiffusionService.GenerateAsBytesAsync(promptOptions, schedulerOptions, ProgressCallback(), _cancelationTokenSource.Token);
                if (result == null)
                    return null;

                var image = Utils.CreateBitmap(result);
                if (image == null)
                    return null;

                return new ImageResult
                {
                    Image = image,
                    Prompt = promptOptions.Prompt,
                    NegativePrompt = promptOptions.NegativePrompt,
                    ProcessType = promptOptions.ProcessType,
                    SchedulerType = promptOptions.SchedulerType,
                    SchedulerOptions = schedulerOptions,
                    Elapsed = Stopwatch.GetElapsedTime(timestamp).TotalSeconds
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating image");
                return null;
            }
        }


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