using Microsoft.Win32;
using OnnxStack.Core.Services;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

namespace OnnxStack.UI.UserControls
{
    public partial class VideoInputControl : UserControl, INotifyPropertyChanged
    {
        private readonly IVideoService _videoService;
        private bool _isPlaying = false;

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoInputControl" /> class.
        /// </summary>
        public VideoInputControl()
        {
            if (!DesignerProperties.GetIsInDesignMode(this))
                _videoService = App.GetService<IVideoService>();

            LoadVideoCommand = new AsyncRelayCommand(LoadVideo);
            ClearVideoCommand = new AsyncRelayCommand(ClearVideo);
            InitializeComponent();
        }

        public AsyncRelayCommand LoadVideoCommand { get; }
        public AsyncRelayCommand ClearVideoCommand { get; }

        public VideoInputModel VideoResult
        {
            get { return (VideoInputModel)GetValue(VideoResultProperty); }
            set { SetValue(VideoResultProperty, value); }
        }
        public static readonly DependencyProperty VideoResultProperty =
            DependencyProperty.Register("VideoResult", typeof(VideoInputModel), typeof(VideoInputControl));

        public SchedulerOptionsModel SchedulerOptions
        {
            get { return (SchedulerOptionsModel)GetValue(SchedulerOptionsProperty); }
            set { SetValue(SchedulerOptionsProperty, value); }
        }
        public static readonly DependencyProperty SchedulerOptionsProperty =
            DependencyProperty.Register("SchedulerOptions", typeof(SchedulerOptionsModel), typeof(VideoInputControl));

        public PromptOptionsModel PromptOptions
        {
            get { return (PromptOptionsModel)GetValue(PromptOptionsProperty); }
            set { SetValue(PromptOptionsProperty, value); }
        }
        public static readonly DependencyProperty PromptOptionsProperty =
            DependencyProperty.Register("PromptOptions", typeof(PromptOptionsModel), typeof(VideoInputControl));

        public bool IsGenerating
        {
            get { return (bool)GetValue(IsGeneratingProperty); }
            set { SetValue(IsGeneratingProperty, value); }
        }
        public static readonly DependencyProperty IsGeneratingProperty =
            DependencyProperty.Register("IsGenerating", typeof(bool), typeof(VideoInputControl), new PropertyMetadata((s, e) =>
            {
                if (s is VideoInputControl control && e.NewValue is bool isGenerating)
                {
                    if (!isGenerating)
                        control.IsPreviewVisible = false;
                }
            }));

        public bool HasVideoResult
        {
            get { return (bool)GetValue(HasVideoResultProperty); }
            set { SetValue(HasVideoResultProperty, value); }
        }
        public static readonly DependencyProperty HasVideoResultProperty =
            DependencyProperty.Register("HasVideoResult", typeof(bool), typeof(VideoInputControl));

        public BitmapImage PreviewImage
        {
            get { return (BitmapImage)GetValue(PreviewImageProperty); }
            set { SetValue(PreviewImageProperty, value); }
        }
        public static readonly DependencyProperty PreviewImageProperty =
            DependencyProperty.Register("PreviewImage", typeof(BitmapImage), typeof(VideoInputControl), new PropertyMetadata((s, e) =>
            {
                if (s is VideoInputControl control && !control.IsPreviewVisible)
                    control.IsPreviewVisible = true;
            }));

        private bool _isPreviewVisible;

        public bool IsPreviewVisible
        {
            get { return _isPreviewVisible; }
            set { _isPreviewVisible = value; NotifyPropertyChanged(); }
        }



        /// <summary>
        /// Loads the image.
        /// </summary>
        /// <returns></returns>
        private async Task LoadVideo()
        {
            // Show Dialog
            var openFileDialog = new OpenFileDialog
            {
                Filter = "Video Files|*.mp4;*.avi;*.mkv;*.mov;*.wmv;*.flv;*.gif|All Files|*.*",
                RestoreDirectory = true,
                Multiselect = false,
            };
            if (openFileDialog.ShowDialog() == true)
            {
                var videoBytes = await File.ReadAllBytesAsync(openFileDialog.FileName);
                var videoInfo = await _videoService.GetVideoInfoAsync(videoBytes);
                VideoResult = new VideoInputModel
                {
                    FileName = openFileDialog.FileName,
                    VideoInfo = videoInfo,
                    VideoBytes = videoBytes
                };
                HasVideoResult = true;
                PromptOptions.VideoInputFPS = videoInfo.FPS;
                PromptOptions.VideoOutputFPS = videoInfo.FPS;
            }
        }


        /// <summary>
        /// Clears the image.
        /// </summary>
        /// <returns></returns>
        private Task ClearVideo()
        {
            VideoResult = null;
            HasVideoResult = false;
            return Task.CompletedTask;
        }


        /// <summary>
        /// Handles the Loaded event of the MediaElement control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="RoutedEventArgs"/> instance containing the event data.</param>
        private void MediaElement_Loaded(object sender, RoutedEventArgs e)
        {
            (sender as MediaElement).Play();
            _isPlaying = true;
        }


        /// <summary>
        /// Handles the MediaEnded event of the MediaElement control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="RoutedEventArgs"/> instance containing the event data.</param>
        private void MediaElement_MediaEnded(object sender, RoutedEventArgs e)
        {
            (sender as MediaElement).Position = TimeSpan.FromMilliseconds(1);
        }


        /// <summary>
        /// Handles the MouseDown event of the MediaElement control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.Windows.Input.MouseButtonEventArgs"/> instance containing the event data.</param>
        private void MediaElement_MouseDown(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            if (sender is not MediaElement mediaElement)
                return;

            if (_isPlaying)
            {
                _isPlaying = false;
                mediaElement.Pause();
                return;
            }

            mediaElement.Play();
            _isPlaying = true;
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
