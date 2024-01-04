using Microsoft.Extensions.Logging;
using OnnxStack.Core.Image;
using OnnxStack.Core.Services;
using OnnxStack.Core.Video;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
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
using System.Windows.Media.Imaging;
using System.Windows.Threading;

namespace OnnxStack.UI.Views
{
    /// <summary>
    /// Interaction logic for VideoToVideoView.xaml
    /// </summary>
    public partial class VideoToVideoView : UserControl, INavigatable, INotifyPropertyChanged
    {
        private readonly ILogger<VideoToVideoView> _logger;
        private readonly IStableDiffusionService _stableDiffusionService;
        private readonly IVideoService _videoService;

        private bool _hasResult;
        private int _progressMax;
        private int _progressValue;
        private string _progressText;
        private bool _isGenerating;
        private int _selectedTabIndex;
        private bool _hasInputResult;
        private bool _isControlsEnabled;
        private VideoInputModel _inputVideo;
        private VideoInputModel _resultVideo;
        private StableDiffusionModelSetViewModel _selectedModel;
        private PromptOptionsModel _promptOptionsModel;
        private SchedulerOptionsModel _schedulerOptions;
        private CancellationTokenSource _cancelationTokenSource;
        private VideoFrames _videoFrames;
        private BitmapImage _previewSource;
        private BitmapImage _previewResult;

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoToVideoView"/> class.
        /// </summary>
        public VideoToVideoView()
        {
            if (!DesignerProperties.GetIsInDesignMode(this))
            {
                _logger = App.GetService<ILogger<VideoToVideoView>>();
                _videoService = App.GetService<IVideoService>();
                _stableDiffusionService = App.GetService<IStableDiffusionService>();
            }

            SupportedDiffusers = new() { DiffuserType.ImageToImage };
            CancelCommand = new AsyncRelayCommand(Cancel, CanExecuteCancel);
            GenerateCommand = new AsyncRelayCommand(Generate, CanExecuteGenerate);
            ClearHistoryCommand = new AsyncRelayCommand(ClearHistory, CanExecuteClearHistory);
            PromptOptions = new PromptOptionsModel();
            SchedulerOptions = new SchedulerOptionsModel();
            ImageResults = new ObservableCollection<ImageResult>();
            ProgressMax = SchedulerOptions.InferenceSteps;
            IsControlsEnabled = true;
            InitializeComponent();
        }

        public OnnxStackUIConfig UISettings
        {
            get { return (OnnxStackUIConfig)GetValue(UISettingsProperty); }
            set { SetValue(UISettingsProperty, value); }
        }
        public static readonly DependencyProperty UISettingsProperty =
            DependencyProperty.Register("UISettings", typeof(OnnxStackUIConfig), typeof(VideoToVideoView));

        public List<DiffuserType> SupportedDiffusers { get; }
        public AsyncRelayCommand CancelCommand { get; }
        public AsyncRelayCommand GenerateCommand { get; }
        public AsyncRelayCommand ClearHistoryCommand { get; set; }
        public ObservableCollection<ImageResult> ImageResults { get; }

        public StableDiffusionModelSetViewModel SelectedModel
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

        public VideoInputModel ResultVideo
        {
            get { return _resultVideo; }
            set { _resultVideo = value; NotifyPropertyChanged(); }
        }

        public VideoInputModel InputVideo
        {
            get { return _inputVideo; }
            set { _inputVideo = value; _videoFrames = null; NotifyPropertyChanged(); }
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

        public string ProgressText
        {
            get { return _progressText; }
            set { _progressText = value; NotifyPropertyChanged(); }
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

        public bool IsControlsEnabled
        {
            get { return _isControlsEnabled; }
            set { _isControlsEnabled = value; NotifyPropertyChanged(); }
        }

        public BitmapImage PreviewSource
        {
            get { return _previewSource; }
            set { _previewSource = value; NotifyPropertyChanged(); }
        }

        public BitmapImage PreviewResult
        {
            get { return _previewResult; }
            set { _previewResult = value; NotifyPropertyChanged(); }
        }


        /// <summary>
        /// Called on Navigate
        /// </summary>
        /// <param name="imageResult">The image result.</param>
        /// <returns></returns>
        public async Task NavigateAsync(ImageResult imageResult)
        {
            throw new NotImplementedException();
        }



        /// <summary>
        /// Generates this image result.
        /// </summary>
        private async Task Generate()
        {
            HasResult = false;
            IsGenerating = true;
            IsControlsEnabled = false;
            ResultVideo = null;
            ProgressMax = 0;
            _cancelationTokenSource = new CancellationTokenSource();

            try
            {
                var schedulerOptions = SchedulerOptions.ToSchedulerOptions();
                if (_videoFrames is null || _videoFrames.Info.FPS != PromptOptions.VideoInputFPS)
                {
                    ProgressText = $"Generating video frames @ {PromptOptions.VideoInputFPS}fps";
                    _videoFrames = await _videoService.CreateFramesAsync(_inputVideo.VideoBytes, PromptOptions.VideoInputFPS, _cancelationTokenSource.Token);
                }
                var promptOptions = GetPromptOptions(PromptOptions, _videoFrames);

                var timestamp = Stopwatch.GetTimestamp();
                var result = await _stableDiffusionService.GenerateAsBytesAsync(new ModelOptions(_selectedModel.ModelSet), promptOptions, schedulerOptions, ProgressCallback(), _cancelationTokenSource.Token);
                var resultVideo = await GenerateResultAsync(result, promptOptions, schedulerOptions, timestamp);
                if (resultVideo != null)
                {
                    App.UIInvoke(() =>
                    {
                        ResultVideo = resultVideo;
                        HasResult = true;
                    });
                }
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation($"Generate was canceled.");
            }
            catch (Exception ex)
            {
                _logger.LogError($"Error during Generate\n{ex}");
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
            return !IsGenerating && HasInputResult;
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
            PreviewSource = null;
            PreviewResult = null;
            IsGenerating = false;
            IsControlsEnabled = true;
            ProgressValue = 0;
            ProgressMax = 1;
            ProgressText = null;
        }


        private PromptOptions GetPromptOptions(PromptOptionsModel promptOptionsModel, VideoFrames videoFrames)
        {
            return new PromptOptions
            {
                Prompt = promptOptionsModel.Prompt,
                NegativePrompt = promptOptionsModel.NegativePrompt,
                DiffuserType = DiffuserType.ImageToImage,
                InputVideo = new VideoInput(videoFrames),
                VideoInputFPS = promptOptionsModel.VideoInputFPS,
                VideoOutputFPS = promptOptionsModel.VideoOutputFPS,
            };
        }


        /// <summary>
        /// Generates the result.
        /// </summary>
        /// <param name="imageBytes">The image bytes.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="timestamp">The timestamp.</param>
        /// <returns></returns>
        private async Task<VideoInputModel> GenerateResultAsync(byte[] videoBytes, PromptOptions promptOptions, SchedulerOptions schedulerOptions, long timestamp)
        {
            var tempVideoFile = Path.Combine(".temp", $"VideoToVideo.mp4");
            await File.WriteAllBytesAsync(tempVideoFile, videoBytes);
            var videoInfo = await _videoService.GetVideoInfoAsync(videoBytes);
            var videoResult = new VideoInputModel
            {
                FileName = tempVideoFile,
                VideoInfo = videoInfo,
                VideoBytes = videoBytes
            };
            return videoResult;
        }


        /// <summary>
        /// StableDiffusion progress callback.
        /// </summary>
        /// <returns></returns>
        private Action<DiffusionProgress> ProgressCallback()
        {
            return (progress) =>
            {
                if (_cancelationTokenSource.IsCancellationRequested)
                    return;

                App.UIInvoke(() =>
                {
                    if (_cancelationTokenSource.IsCancellationRequested)
                        return;

                    if (progress.BatchTensor is not null)
                    {
                        PreviewResult = Utils.CreateBitmap(progress.BatchTensor.ToImageBytes());
                        PreviewSource = Utils.CreateBitmap(_videoFrames.Frames[progress.BatchValue - 1]);
                        ProgressText = $"Video Frame {progress.BatchValue} of {_videoFrames.Frames.Count} complete";
                    }

                    if (ProgressText != progress.Message && progress.BatchMax == 0)
                        ProgressText = progress.Message;

                    if (ProgressMax != progress.BatchMax)
                        ProgressMax = progress.BatchMax;

                    if (ProgressValue != progress.BatchValue)
                        ProgressValue = progress.BatchValue;

                }, DispatcherPriority.Background);
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