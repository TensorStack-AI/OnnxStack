using LibGit2Sharp;
using Microsoft.Extensions.Logging;
using Models;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
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
        private ModelOptionsModel _selectedModel;
        private PromptOptionsModel _promptOptionsModel;
        private SchedulerOptionsModel _schedulerOptions;
        private BatchOptionsModel _batchOptions;
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

            SupportedDiffusers = new() { DiffuserType.ImageToImage };
            CancelCommand = new AsyncRelayCommand(Cancel, CanExecuteCancel);
            GenerateCommand = new AsyncRelayCommand(Generate, CanExecuteGenerate);
            ClearHistoryCommand = new AsyncRelayCommand(ClearHistory, CanExecuteClearHistory);
            PromptOptions = new PromptOptionsModel();
            SchedulerOptions = new SchedulerOptionsModel();
            BatchOptions = new BatchOptionsModel();
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
            DependencyProperty.Register("UISettings", typeof(OnnxStackUIConfig), typeof(ImageToImage));

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

        public BatchOptionsModel BatchOptions
        {
            get { return _batchOptions; }
            set { _batchOptions = value; NotifyPropertyChanged(); }
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
            if (imageResult.Model.ModelOptions.Diffusers.Contains(DiffuserType.ImageToImage))
            {
                SelectedModel = imageResult.Model;
            }
            InputImage = new ImageInput
            {
                Image = imageResult.Image,
                FileName = "OnnxStack Generated Image"
            };
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
                DiffuserType = DiffuserType.ImageToImage,
                InputImage = new StableDiffusion.Models.InputImage
                {
                    ImageBytes = InputImage.Image.GetImageBytes()
                }
            };

            var batchOptions = BatchOptions.ToBatchOptions();
            var schedulerOptions = SchedulerOptions.ToSchedulerOptions();

            try
            {
                await foreach (var resultImage in ExecuteStableDiffusion(_selectedModel.ModelOptions, promptOptions, schedulerOptions, batchOptions))
                {
                    if (resultImage != null)
                    {
                        ResultImage = resultImage;
                        ImageResults.Add(resultImage);
                        HasResult = true;
                    }
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
            return !IsGenerating
                && !string.IsNullOrEmpty(PromptOptions.Prompt) 
                && HasInputResult;
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
        /// <param name="modelOptions">The model options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="batchOptions">The batch options.</param>
        /// <returns></returns>
        private async IAsyncEnumerable<ImageResult> ExecuteStableDiffusion(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions)
        {
            _cancelationTokenSource = new CancellationTokenSource();
            if (!BatchOptions.IsAutomationEnabled)
            {
                var timestamp = Stopwatch.GetTimestamp();
                var result = await _stableDiffusionService.GenerateAsBytesAsync(modelOptions, promptOptions, schedulerOptions, ProgressCallback(), _cancelationTokenSource.Token);
                yield return await GenerateResult(result, promptOptions, schedulerOptions, timestamp);
            }
            else
            {
                await foreach (var batchResult in _stableDiffusionService.GenerateBatchAsync(modelOptions, promptOptions, schedulerOptions, batchOptions, ProgressBatchCallback(), _cancelationTokenSource.Token))
                {
                    var timestamp = Stopwatch.GetTimestamp();
                    yield return await GenerateResult(batchResult.ImageResult.ToImageBytes(), promptOptions, batchResult.SchedulerOptions, timestamp);
                }
            }
        }


        /// <summary>
        /// Generates the result.
        /// </summary>
        /// <param name="imageBytes">The image bytes.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="timestamp">The timestamp.</param>
        /// <returns></returns>
        private async Task<ImageResult> GenerateResult(byte[] imageBytes, PromptOptions promptOptions, SchedulerOptions schedulerOptions, long timestamp)
        {
            var image = Utils.CreateBitmap(imageBytes);

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
                await imageResult.AutoSave(Path.Combine(UISettings.ImageAutoSaveDirectory, "ImageToImage"), UISettings.ImageAutoSaveBlueprint);
            return imageResult;
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

        private Action<int, int, int, int> ProgressBatchCallback()
        {
            return (batchIndex, batchCount, step, steps) =>
            {
                App.UIInvoke(() =>
                {
                    if (_cancelationTokenSource.IsCancellationRequested)
                        return;

                    if (BatchOptions.BatchsValue != batchCount)
                        BatchOptions.BatchsValue = batchCount;
                    if (BatchOptions.BatchValue != batchIndex)
                        BatchOptions.BatchValue = batchIndex;
                    if (BatchOptions.StepValue != step)
                        BatchOptions.StepValue = step;
                    if (BatchOptions.StepsValue != steps)
                        BatchOptions.StepsValue = steps;
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