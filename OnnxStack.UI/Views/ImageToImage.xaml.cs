using Microsoft.Extensions.Logging;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
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
    /// Interaction logic for ImageToImage.xaml
    /// </summary>
    public partial class ImageToImage : UserControl, INavigatable, INotifyPropertyChanged
    {
        private readonly ILogger<TextToImageView> _logger;
        private readonly IStableDiffusionService _stableDiffusionService;

        private bool _hasResult;
        private int _progressMax;
        private int _progressValue;
        private bool _isGenerating;
        private int _selectedTabIndex;
        private bool _hasInputResult;
        private ImageInput _inputImage;
        private ImageResult _resultImage;
        private PromptOptionsModel _promptOptionsModel;
        private SchedulerOptionsModel _schedulerOptions;
        private CancellationTokenSource _cancelationTokenSource;


        /// <summary>
        /// Initializes a new instance of the <see cref="ImageToImage"/> class.
        /// </summary>
        public ImageToImage()
        {
            if (!DesignerProperties.GetIsInDesignMode(this))
            {
                _logger = App.GetService<ILogger<TextToImageView>>();
                _stableDiffusionService = App.GetService<IStableDiffusionService>();
            }
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

        public ImageInput InputImage
        {
            get { return _inputImage; }
            set { _inputImage = value; NotifyPropertyChanged(); }
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

        public bool HasInputResult
        {
            get { return _hasInputResult; }
            set { _hasInputResult = value; NotifyPropertyChanged(); }
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
            HasInputResult = true;
            InputImage = new ImageInput
            {
                Image = imageResult.Image,
                FileName = "OnnxStack Generated Image"
            };
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
                SchedulerType = PromptOptions.SchedulerType,
                ProcessType = ProcessType.ImageToImage,
                InputImage = new StableDiffusion.Models.InputImage
                {
                    ImageBytes = InputImage.Image.GetImageBytes()
                }
            };

            var schedulerOptions = SchedulerOptions.ToSchedulerOptions();
            var resultImage = await ExecuteStableDiffusion(promptOptions, schedulerOptions);
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
            return !IsGenerating && !string.IsNullOrEmpty(PromptOptions.Prompt) && HasInputResult;
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
        /// Executes the stable diffusion process.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <returns></returns>
        private async Task<ImageResult> ExecuteStableDiffusion(PromptOptions promptOptions, SchedulerOptions schedulerOptions)
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