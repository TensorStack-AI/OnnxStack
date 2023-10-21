using Microsoft.Extensions.Logging;
using OnnxStack.UI.Commands;
using System;
using System.ComponentModel;
using System.IO;
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

        private int _zoom = 100;
        private double _scale = 1.0;
        private int _maxWidth = 1024;
        private int _maxHeight = 512;
        private bool _isCropped;
        private int _cropWidth;
        private int _cropHeight;
        private double _imageWidth;
        private double _imageHeight;
        private double _zoomX;
        private double _zoomY;
        private double _zoomWidth;
        private double _zoomHeight;

        private string _imageFile;
        private BitmapSource _sourceImage;
        private BitmapSource _resultImage;
        private bool _cropIsDragging;
        private Point _cropClickPosition;
        private TranslateTransform _cropTransform;

        public CropImageDialog(ILogger<CropImageDialog> logger)
        {
            _logger = logger;
            OkCommand = new AsyncRelayCommand(Ok, CanExecuteOk);
            CancelCommand = new AsyncRelayCommand(Cancel, CanExecuteCancel);
            CropCommand = new AsyncRelayCommand(Crop, CanExecuteCrop);
            InitializeComponent();
        }

        public AsyncRelayCommand OkCommand { get; }
        public AsyncRelayCommand CancelCommand { get; }
        public AsyncRelayCommand CropCommand { get; set; }

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

        public int CropWidth
        {
            get { return _cropWidth; }
            set { _cropWidth = value; NotifyPropertyChanged(); }
        }

        public int CropHeight
        {
            get { return _cropHeight; }
            set { _cropHeight = value; NotifyPropertyChanged(); }
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

        public void Initialize(int width, int height)
        {
            CropWidth = width;
            CropHeight = height;
        }

        public BitmapSource GetImage()
        {
            return _resultImage?.Clone();
        }

        public byte[] GetImageBytes()
        {
            if (_resultImage == null)
                return null;

            using (var stream = new MemoryStream())
            {
                var encoder = new PngBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(_resultImage));
                encoder.Save(stream);
                return stream.ToArray();
            }
        }


        private Task Ok()
        {
            DialogResult = true;
            return Task.CompletedTask;
        }


        private bool CanExecuteOk()
        {
            return IsCropped;
        }

        private Task Cancel()
        {
            DialogResult = false;
            return Task.CompletedTask;
        }


        private bool CanExecuteCancel()
        {
            return true;
        }

        private bool CanExecuteCrop()
        {
            return !string.IsNullOrEmpty(ImageFile) && !IsCropped;
        }

        private Task Crop()
        {
            _resultImage = CropAndResizeImage();
            Reset();
            IsCropped = true;
            ImageWidth = CropWidth;
            ImageHeight = CropHeight;
            ZoomWidth = CropWidth;
            ZoomHeight = CropHeight;
            SourceImage = _resultImage;
            // Set ResultImage to the Image control
            return Task.CompletedTask;
        }


        private void LoadImage()
        {
            Reset();
            SourceImage = new BitmapImage(new Uri(ImageFile));
            var actualWidth = _sourceImage.Width;
            var actualHeight = _sourceImage.Height;
            if (actualWidth > _maxWidth || actualHeight > _maxHeight)
            {
                _scale = Math.Min(1, actualWidth > actualHeight
                    ? (_maxHeight / actualHeight)
                    : (_maxWidth / actualWidth));
            }

            ImageWidth = actualWidth * _scale;
            ImageHeight = actualHeight * _scale;
            ZoomWidth = CropWidth;
            ZoomHeight = CropHeight;
            HandleZoom();
        }

        private void Reset()
        {
            _zoom = 100;
            _zoomX = 0;
            _zoomY = 0;
            _scale = 1;
            IsCropped = false;
            CropFrame.RenderTransform = new TranslateTransform(0, 0);
        }

        private BitmapSource CropAndResizeImage()
        {
            var zoom = _zoom / 100.0;
            var rect = new Int32Rect
            {
                X = (int)(_zoomX / _scale),
                Y = (int)(_zoomY / _scale),
                Width = (int)((CropWidth * zoom) / _scale),
                Height = (int)((CropHeight * zoom) / _scale)
            };

            var croppedBitmap = new CroppedBitmap(_sourceImage, rect);
            var scaleX = (double)CropWidth / croppedBitmap.PixelWidth;
            var scaleY = (double)CropHeight / croppedBitmap.PixelHeight;
            var scaleTransform = new ScaleTransform(scaleX, scaleY);
            return new TransformedBitmap(croppedBitmap, scaleTransform);
        }


        private void HandleZoom(int delta = 1)
        {
            _zoom = delta > 0
                ? Math.Min(++_zoom, 100)
                : Math.Max(--_zoom, -90);
            ZoomWidth = (CropWidth / 100.0) * _zoom;
            ZoomHeight = (CropHeight / 100.0) * _zoom;
        }


        private void Canvas_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
             _cropTransform = CropFrame.RenderTransform as TranslateTransform ?? new TranslateTransform();
            _cropIsDragging = true;
            _cropClickPosition = e.GetPosition(this);
            CropFrame.CaptureMouse();
        }

        private void Canvas_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            _cropIsDragging = false;
            CropFrame.ReleaseMouseCapture();
        }

        private void Canvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (!_cropIsDragging)
                return;

            Point currentPosition = e.GetPosition(this);
            var transform = CropFrame.RenderTransform as TranslateTransform ?? new TranslateTransform();
            var x = _cropTransform.X + (currentPosition.X - _cropClickPosition.X);
            var y = _cropTransform.Y + (currentPosition.Y - _cropClickPosition.Y);
            _zoomX = Math.Max(0, Math.Min(x, ImageWidth - ZoomWidth));
            _zoomY = Math.Max(0, Math.Min(y, ImageHeight - ZoomHeight));

            transform.X = _zoomX;
            transform.Y = _zoomY;
            CropFrame.RenderTransform = new TranslateTransform(transform.X, transform.Y);
        }

        private void Canvas_MouseWheel(object sender, MouseWheelEventArgs e)
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
