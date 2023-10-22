using Microsoft.Extensions.Logging;
using OnnxStack.UI.Commands;
using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace OnnxStack.UI.Dialogs
{
    /// <summary>
    /// Interaction logic for CropImageDialog.xaml
    /// </summary>
    public partial class CropImageDialog : Window, INotifyPropertyChanged
    {
        private readonly ILogger<CropImageDialog> _logger;
        private double _zoom = 100;
        private double _scale = 1.0;
        private int _maxWidth = 960;
        private int _maxHeight = 448;
        private bool _isCropped;
        private double _cropWidth;
        private double _cropHeight;
        private double _imageWidth = 400;
        private double _imageHeight = 400;
        private double _zoomX;
        private double _zoomY;
        private double _zoomWidth;
        private double _zoomHeight;
        private int _requiredWidth;
        private int _requiredHeight;
        private string _imageFile;
        private BitmapSource _sourceImage;
        private BitmapSource _resultImage;
        private bool _cropIsDragging;
        private Point _cropClickPosition;
        private TranslateTransform _cropTransform;


        /// <summary>
        /// Initializes a new instance of the <see cref="CropImageDialog"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        public CropImageDialog(ILogger<CropImageDialog> logger)
        {
            _logger = logger;
            DoneCommand = new AsyncRelayCommand(Done, CanExecuteDone);
            CancelCommand = new AsyncRelayCommand(Cancel, CanExecuteCancel);
            CropCommand = new AsyncRelayCommand(Crop, CanExecuteCrop);
            ResetCommand = new AsyncRelayCommand(ResetSource);
            InitializeComponent();
        }

        public AsyncRelayCommand DoneCommand { get; }
        public AsyncRelayCommand CancelCommand { get; }
        public AsyncRelayCommand CropCommand { get; }
        public AsyncRelayCommand ResetCommand { get; }

        public BitmapSource SourceImage
        {
            get { return _sourceImage; }
            set { _sourceImage = value; NotifyPropertyChanged(); }
        }

        public string ImageFile
        {
            get { return _imageFile; }
            set { _imageFile = value; NotifyPropertyChanged(); LoadImage(); }
        }

        public double ImageWidth
        {
            get { return _imageWidth; }
            set { _imageWidth = value; NotifyPropertyChanged(); }
        }

        public double ImageHeight
        {
            get { return _imageHeight; }
            set { _imageHeight = value; NotifyPropertyChanged(); }
        }

        public double ZoomWidth
        {
            get { return _zoomWidth; }
            set { _zoomWidth = value; NotifyPropertyChanged(); }
        }

        public double ZoomHeight
        {
            get { return _zoomHeight; }
            set { _zoomHeight = value; NotifyPropertyChanged(); }
        }

        public bool IsCropped
        {
            get { return _isCropped; }
            set { _isCropped = value; NotifyPropertyChanged(); }
        }


        /// <summary>
        /// Initializes the specified the Dialog.
        /// </summary>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        public void Initialize(int requiredWidth, int requiredHeight)
        {
            _requiredWidth = requiredWidth;
            _requiredHeight = requiredHeight;
        }


        /// <summary>
        /// Gets the image.
        /// </summary>
        /// <returns></returns>
        public BitmapSource GetImageResult()
        {
            return _resultImage?.Clone();
        }


        /// <summary>
        /// Called when the user is finished
        /// </summary>
        /// <returns></returns>
        private Task Done()
        {
            DialogResult = true;
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can execute Done.
        /// </summary>
        /// <returns>
        ///   <c>true</c> if this instance can execute Done; otherwise, <c>false</c>.
        /// </returns>
        private bool CanExecuteDone()
        {
            return IsCropped;
        }


        /// <summary>
        /// Cancels this instance.
        /// </summary>
        /// <returns></returns>
        private Task Cancel()
        {
            DialogResult = false;
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can execute cancel.
        /// </summary>
        /// <returns>
        ///   <c>true</c> if this instance can execute cancel; otherwise, <c>false</c>.
        /// </returns>
        private bool CanExecuteCancel()
        {
            return true;
        }


        /// <summary>
        /// Crops this SourceImage.
        /// </summary>
        /// <returns></returns>
        private Task Crop()
        {
            _resultImage = CropAndResizeImage();
            Reset();
            IsCropped = true;
            ImageWidth = _cropWidth;
            ImageHeight = _cropHeight;
            ZoomWidth = _cropWidth;
            ZoomHeight = _cropHeight;
            SourceImage = _resultImage;
            // Set ResultImage to the Image control
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can execute Crop.
        /// </summary>
        /// <returns>
        ///   <c>true</c> if this instance can execute Crop; otherwise, <c>false</c>.
        /// </returns>
        private bool CanExecuteCrop()
        {
            return !string.IsNullOrEmpty(ImageFile) && !IsCropped;
        }


        /// <summary>
        /// Loads the image.
        /// </summary>
        private void LoadImage()
        {
            Reset();
            SourceImage = new BitmapImage(new Uri(ImageFile));
            var actualWidth = (double)_sourceImage.PixelWidth;
            var actualHeight = (double)_sourceImage.PixelHeight;

            // Scale Image
            double scaleX = _maxWidth / actualWidth;
            double scaleY = _maxHeight / actualHeight;
            _scale = Math.Min(scaleX, scaleY);
            ImageWidth = actualWidth * _scale;
            ImageHeight = actualHeight * _scale;

            // Scale Crop Rectangle
            var cropScaleX = ImageWidth / _requiredWidth;
            var cropScaleY = ImageHeight / _requiredHeight;
            var cropScale = Math.Min(cropScaleX, cropScaleY);
            _cropWidth = _requiredWidth * cropScale;
            _cropHeight = _requiredHeight * cropScale;

            ZoomWidth = _cropWidth;
            ZoomHeight = _cropHeight;
            CropFrame.RenderTransform = new TranslateTransform((ImageWidth - ZoomWidth) / 2, (ImageHeight - ZoomHeight) / 2);
            HandleZoom();
        }


        /// <summary>
        /// Resets this instance.
        /// </summary>
        private void Reset()
        {
            _zoom = 96;
            _zoomX = 0;
            _zoomY = 0;
            _scale = 1;
            IsCropped = false;
            _cropTransform = new TranslateTransform(0, 0);
            CropFrame.RenderTransform = new TranslateTransform(0, 0);
        }


        /// <summary>
        /// Resets the source image.
        /// </summary>
        /// <returns></returns>
        private Task ResetSource()
        {
            LoadImage();
            return Task.CompletedTask;
        }


        /// <summary>
        /// Crops and resize image.
        /// </summary>
        /// <returns></returns>
        private BitmapSource CropAndResizeImage()
        {
            var zoom = _zoom / 100.0;
            var rect = new Int32Rect
            {
                X = (int)Math.Max(_zoomX / _scale, 0),
                Y = (int)Math.Max(_zoomY / _scale, 0),
                Width = (int)Math.Min((_cropWidth * zoom) / _scale, _sourceImage.PixelWidth),
                Height = (int)Math.Min((_cropHeight * zoom) / _scale, _sourceImage.PixelHeight)
            };

            try
            {
                var croppedBitmap = new CroppedBitmap(_sourceImage, rect);
                var scaleX = _cropWidth / croppedBitmap.PixelWidth;
                var scaleY = _cropHeight / croppedBitmap.PixelHeight;
                var scaleTransform = new ScaleTransform(scaleX, scaleY);
                return new TransformedBitmap(croppedBitmap, scaleTransform);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error cropping image");
                DialogResult = false;
                return null;
            }
        }


        /// <summary>
        /// Gets the crop transfrom.
        /// </summary>
        /// <returns></returns>
        private TranslateTransform GetCropTransfrom()
        {
            return CropFrame.RenderTransform as TranslateTransform;
        }


        /// <summary>
        /// Handles the zoom.
        /// </summary>
        /// <param name="delta">The delta.</param>
        private void HandleZoom(int delta = 1)
        {
            var isZoomIn = delta > 0;
            var currentZoom = _zoom;
            var newZoom = isZoomIn
                ? Math.Min(++currentZoom, 500)
                : Math.Max(--currentZoom, -90);

            var transform = GetCropTransfrom();
            var zoomWidth = (_cropWidth / 100.0) * newZoom;
            var zoomHeight = (_cropHeight / 100.0) * newZoom;
            var zoomX = transform.X + ((ZoomWidth - zoomWidth) / 2);
            var zoomY = transform.Y + ((ZoomHeight - zoomHeight) / 2);
            var outOfBounds = zoomX + CropFrame.ActualWidth > ImageWidth || zoomY + CropFrame.ActualHeight > ImageHeight || transform.X < 0 || transform.Y < 0;
            if (isZoomIn && outOfBounds)
                return;

            _zoomX = zoomX;
            _zoomY = zoomY;
            _zoom = newZoom;
            ZoomWidth = zoomWidth;
            ZoomHeight = zoomHeight;
            CropFrame.RenderTransform = new TranslateTransform(_zoomX, _zoomY);
        }


        /// <summary>
        /// Handles the MouseLeftButtonDown event of the CropFrame control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="MouseButtonEventArgs"/> instance containing the event data.</param>
        private void CropFrame_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            _cropTransform = GetCropTransfrom();
            _cropIsDragging = true;
            _cropClickPosition = e.GetPosition(this);
            CropFrame.CaptureMouse();
        }


        /// <summary>
        /// Handles the MouseLeftButtonUp event of the CropFrame control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="MouseButtonEventArgs"/> instance containing the event data.</param>
        private void CropFrame_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            _cropIsDragging = false;
            CropFrame.ReleaseMouseCapture();
        }



        /// <summary>
        /// Handles the MouseMove event of the CropFrame control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="MouseEventArgs"/> instance containing the event data.</param>
        private void CropFrame_MouseMove(object sender, MouseEventArgs e)
        {
            if (!_cropIsDragging)
                return;

            Point currentPosition = e.GetPosition(this);
            var x = _cropTransform.X + (currentPosition.X - _cropClickPosition.X);
            var y = _cropTransform.Y + (currentPosition.Y - _cropClickPosition.Y);
            _zoomX = Math.Max(0, Math.Min(x, ImageWidth - ZoomWidth));
            _zoomY = Math.Max(0, Math.Min(y, ImageHeight - ZoomHeight));
            CropFrame.RenderTransform = new TranslateTransform(_zoomX, _zoomY);
        }


        /// <summary>
        /// Handles the MouseWheel event of the CropFrame control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="MouseWheelEventArgs"/> instance containing the event data.</param>
        private void CropFrame_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            HandleZoom(e.Delta);
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
