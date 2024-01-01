using Microsoft.Extensions.Logging;
using Microsoft.Win32;
using OnnxStack.StableDiffusion.Config;
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
    public partial class VideoResultControl : UserControl, INotifyPropertyChanged
    {
        private readonly ILogger<VideoResultControl> _logger;
        private bool _isPlaying = false;

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoResultControl" /> class.
        /// </summary>
        public VideoResultControl()
        {
            if (!DesignerProperties.GetIsInDesignMode(this))
                _logger = App.GetService<ILogger<VideoResultControl>>();

            SaveVideoCommand = new AsyncRelayCommand(SaveVideo);
            ClearVideoCommand = new AsyncRelayCommand(ClearVideo);
            InitializeComponent();
            HasVideoResult = false;
        }

        public AsyncRelayCommand SaveVideoCommand { get; }
        public AsyncRelayCommand ClearVideoCommand { get; }

        public VideoInputModel VideoResult
        {
            get { return (VideoInputModel)GetValue(VideoResultProperty); }
            set { SetValue(VideoResultProperty, value); }
        }
        public static readonly DependencyProperty VideoResultProperty =
            DependencyProperty.Register("VideoResult", typeof(VideoInputModel), typeof(VideoResultControl));

        public SchedulerOptionsModel SchedulerOptions
        {
            get { return (SchedulerOptionsModel)GetValue(SchedulerOptionsProperty); }
            set { SetValue(SchedulerOptionsProperty, value); }
        }
        public static readonly DependencyProperty SchedulerOptionsProperty =
            DependencyProperty.Register("SchedulerOptions", typeof(SchedulerOptionsModel), typeof(VideoResultControl));

        public bool HasVideoResult
        {
            get { return (bool)GetValue(HasVideoResultProperty); }
            set { SetValue(HasVideoResultProperty, value); }
        }
        public static readonly DependencyProperty HasVideoResultProperty =
            DependencyProperty.Register("HasVideoResult", typeof(bool), typeof(VideoResultControl));

        public int ProgressMax
        {
            get { return (int)GetValue(ProgressMaxProperty); }
            set { SetValue(ProgressMaxProperty, value); }
        }
        public static readonly DependencyProperty ProgressMaxProperty =
            DependencyProperty.Register("ProgressMax", typeof(int), typeof(VideoResultControl));

        public int ProgressValue
        {
            get { return (int)GetValue(ProgressValueProperty); }
            set { SetValue(ProgressValueProperty, value); }
        }
        public static readonly DependencyProperty ProgressValueProperty =
            DependencyProperty.Register("ProgressValue", typeof(int), typeof(VideoResultControl));

        public string ProgressText
        {
            get { return (string)GetValue(ProgressTextProperty); }
            set { SetValue(ProgressTextProperty, value); }
        }
        public static readonly DependencyProperty ProgressTextProperty =
            DependencyProperty.Register("ProgressText", typeof(string), typeof(VideoResultControl));

        public BitmapImage PreviewImage
        {
            get { return (BitmapImage)GetValue(PreviewImageProperty); }
            set { SetValue(PreviewImageProperty, value); }
        }
        public static readonly DependencyProperty PreviewImageProperty =
            DependencyProperty.Register("PreviewImage", typeof(BitmapImage), typeof(VideoResultControl));



        /// <summary>
        /// Saves the video.
        /// </summary>
        private async Task SaveVideo()
        {
            await SaveVideoFile(VideoResult, SchedulerOptions);
        }

        /// <summary>
        /// Clears the image.
        /// </summary>
        /// <returns></returns>
        private Task ClearVideo()
        {
            ProgressMax = 1;
            VideoResult = null;
            HasVideoResult = false;
            return Task.CompletedTask;
        }


        /// <summary>
        /// Saves the video file.
        /// </summary>
        /// <param name="videoResult">The video result.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        private async Task SaveVideoFile(VideoInputModel videoResult, SchedulerOptionsModel schedulerOptions)
        {
            try
            {
                var saveFileDialog = new SaveFileDialog
                {
                    Filter = "mp4 files (*.mp4)|*.mp4",
                    DefaultExt = "mp4",
                    AddExtension = true,
                    RestoreDirectory = true,
                    InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyPictures),
                    FileName = $"video-{schedulerOptions.Seed}.mp4"
                };

                var dialogResult = saveFileDialog.ShowDialog();
                if (dialogResult == false)
                {
                    _logger.LogInformation("Saving video canceled");
                    return;
                }

                // Write File
                await File.WriteAllBytesAsync(saveFileDialog.FileName, videoResult.VideoBytes);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error saving video");
            }
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
