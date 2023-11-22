using OnnxStack.Core;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Dialogs;
using OnnxStack.UI.Models;
using OnnxStack.UI.Services;
using System;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Ink;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace OnnxStack.UI.UserControls
{
    public partial class PaintInputControl : UserControl, INotifyPropertyChanged
    {
        private readonly IDialogService _dialogService;

        private int _brushSize;
        private bool _isEraserEnabled;
        public DateTime _canvasLastUpdate;
        private int _canvasUpdatedelay = 250;
        private DrawingAttributes _brushAttributes;
        private Color _selectedColor = Colors.Black;
        private Brush _backgroundBrush = new SolidColorBrush(Colors.White);
        private InkCanvasEditingMode _canvasEditingMode = InkCanvasEditingMode.Ink;

        /// <summary>
        /// Initializes a new instance of the <see cref="PaintInputControl" /> class.
        /// </summary>
        public PaintInputControl()
        {
            if (!DesignerProperties.GetIsInDesignMode(this))
                _dialogService = App.GetService<IDialogService>();

            BrushDrawSize = 20;
            LoadImageCommand = new AsyncRelayCommand(LoadImage);
            ClearImageCommand = new AsyncRelayCommand(ClearImage);
            CanvasModeCommand = new AsyncRelayCommand(CanvasMode);
            CopyImageCommand = new AsyncRelayCommand(CopyImage);
            PasteImageCommand = new AsyncRelayCommand(PasteImage);
            InitializeComponent();
        }

        public AsyncRelayCommand LoadImageCommand { get; }
        public AsyncRelayCommand ClearImageCommand { get; }
        public AsyncRelayCommand CanvasModeCommand { get; }
        public AsyncRelayCommand CopyImageCommand { get; }
        public AsyncRelayCommand PasteImageCommand { get; }

        public ImageInput InputImage
        {
            get { return (ImageInput)GetValue(InputImageProperty); }
            set { SetValue(InputImageProperty, value); }
        }
        public static readonly DependencyProperty InputImageProperty =
            DependencyProperty.Register("InputImage", typeof(ImageInput), typeof(PaintInputControl), new PropertyMetadata(async (s, e) =>
            {
                if (s is PaintInputControl control && e.NewValue is ImageInput image)
                {
                    control.BackgroundBrush = new ImageBrush(image.Image);
                    await Task.Delay(500); // TODO: Fix race condition
                    await control.SaveCanvas();
                }
            }));

        public ImageInput CanvasResult
        {
            get { return (ImageInput)GetValue(CanvasResultProperty); }
            set { SetValue(CanvasResultProperty, value); }
        }
        public static readonly DependencyProperty CanvasResultProperty =
            DependencyProperty.Register("CanvasResult", typeof(ImageInput), typeof(PaintInputControl));


        public SchedulerOptionsModel SchedulerOptions
        {
            get { return (SchedulerOptionsModel)GetValue(SchedulerOptionsProperty); }
            set { SetValue(SchedulerOptionsProperty, value); }
        }
        public static readonly DependencyProperty SchedulerOptionsProperty =
            DependencyProperty.Register("SchedulerOptions", typeof(SchedulerOptionsModel), typeof(PaintInputControl));


        public bool HasCanvasChanged
        {
            get { return (bool)GetValue(HasCanvasChangedProperty); }
            set { SetValue(HasCanvasChangedProperty, value); }
        }
        public static readonly DependencyProperty HasCanvasChangedProperty =
            DependencyProperty.Register("HasCanvasChanged", typeof(bool), typeof(PaintInputControl));


        public InkCanvasEditingMode CanvasEditingMode
        {
            get { return _canvasEditingMode; }
            set { _canvasEditingMode = value; NotifyPropertyChanged(); }
        }

        public DrawingAttributes BrushAttributes
        {
            get { return _brushAttributes; }
            set { _brushAttributes = value; NotifyPropertyChanged(); }
        }

        public bool IsEraserEnabled
        {
            get { return _isEraserEnabled; }
            set { _isEraserEnabled = value; NotifyPropertyChanged(); }
        }

        public int BrushDrawSize
        {
            get { return _brushSize; }
            set
            {
                _brushSize = value;
                NotifyPropertyChanged();
                UpdateBrushAttributes();
            }
        }

        public Color SelectedColor
        {
            get { return _selectedColor; }
            set
            {
                _selectedColor = value;
                NotifyPropertyChanged();
                UpdateBrushAttributes();
            }
        }

        public Brush BackgroundBrush
        {
            get { return _backgroundBrush; }
            set { _backgroundBrush = value; NotifyPropertyChanged(); }
        }



        /// <summary>
        /// Loads the image.
        /// </summary>
        /// <returns></returns>
        private Task LoadImage()
        {
            ShowCropImageDialog();
            return Task.CompletedTask;
        }


        /// <summary>
        /// Clears the image.
        /// </summary>
        /// <returns></returns>
        private Task ClearImage()
        {
            ClearCanvas();
            return Task.CompletedTask;
        }


        /// <summary>
        /// Clears the Canvas.
        /// </summary>
        private void ClearCanvas()
        {
            PaintCanvas.Strokes.Clear();
            CanvasResult = new ImageInput
            {
                Image = CreateEmptyCanvasImage(),
                FileName = "Canvas Image",
            };
            BackgroundBrush = new SolidColorBrush(Colors.White);
            HasCanvasChanged = true;
            IsEraserEnabled = false;
            CanvasEditingMode = InkCanvasEditingMode.Ink;
        }


        /// <summary>
        /// Saves the Canvas
        /// </summary>
        /// <returns></returns>
        private Task SaveCanvas()
        {
            CanvasResult = new ImageInput
            {
                Image = CreateCanvasImage(),
                FileName = "Canvas Image",
            };
            HasCanvasChanged = true;
            return Task.CompletedTask;
        }


        /// <summary>
        /// Change Canvas mode.
        /// </summary>
        /// <returns></returns>
        private Task CanvasMode()
        {
            if (_isEraserEnabled)
            {
                IsEraserEnabled = false;
                CanvasEditingMode = InkCanvasEditingMode.Ink;
            }
            else
            {
                IsEraserEnabled = true;
                CanvasEditingMode = InkCanvasEditingMode.EraseByPoint;
            }
            return Task.CompletedTask;
        }


        /// <summary>
        /// Updates the brush attributes.
        /// </summary>
        private void UpdateBrushAttributes()
        {
            BrushAttributes = new DrawingAttributes
            {
                Color = _selectedColor,
                Height = _brushSize,
                Width = _brushSize,
            };
        }


        /// <summary>
        /// Creates the canvas image.
        /// </summary>
        /// <returns></returns>
        public BitmapSource CreateCanvasImage()
        {
            if (PaintCanvas.ActualWidth == 0)
                return CreateEmptyCanvasImage();

            // Create a RenderTargetBitmap to render the Canvas content.
            var renderBitmap = new RenderTargetBitmap((int)PaintCanvas.ActualWidth, (int)PaintCanvas.ActualHeight, 96, 96, PixelFormats.Pbgra32);

            // Make a drawing visual to render.
            var visual = new DrawingVisual();
            using (DrawingContext context = visual.RenderOpen())
            {
                VisualBrush brush = new VisualBrush(PaintCanvas);
                context.DrawRectangle(brush, null, new Rect(new Point(0, 0), new Point(PaintCanvas.ActualWidth, PaintCanvas.ActualHeight)));
            }
            renderBitmap.Render(visual);
            return renderBitmap;
        }



        /// <summary>
        /// Creates the empty canvas image.
        /// </summary>
        /// <returns></returns>
        public BitmapSource CreateEmptyCanvasImage()
        {
            var wbm = new WriteableBitmap(SchedulerOptions.Width, SchedulerOptions.Height, 96, 96, PixelFormats.Bgra32, null);
            BitmapImage bmImage = new BitmapImage();
            using (MemoryStream stream = new MemoryStream())
            {
                PngBitmapEncoder encoder = new PngBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(wbm));
                encoder.Save(stream);
                bmImage.BeginInit();
                bmImage.CacheOption = BitmapCacheOption.OnLoad;
                bmImage.StreamSource = stream;
                bmImage.EndInit();
                bmImage.Freeze();
            }
            return bmImage;
        }



        /// <summary>
        /// Shows the crop image dialog.
        /// </summary>
        /// <param name="source">The source.</param>
        /// <param name="sourceFile">The source file.</param>
        private async void ShowCropImageDialog(BitmapSource source = null, string sourceFile = null)
        {
            try
            {
                if (!string.IsNullOrEmpty(sourceFile))
                    source = new BitmapImage(new Uri(sourceFile));
            }
            catch { }

            var loadImageDialog = _dialogService.GetDialog<CropImageDialog>();
            loadImageDialog.Initialize(SchedulerOptions.Width, SchedulerOptions.Height, source);
            if (loadImageDialog.ShowDialog() == true)
            {
                BackgroundBrush = new ImageBrush(loadImageDialog.GetImageResult());
                await SaveCanvas();
            }
        }


        /// <summary>
        /// Copies the image.
        /// </summary>
        /// <returns></returns>
        private Task CopyImage()
        {
            if (CanvasResult?.Image != null)
                Clipboard.SetImage(CanvasResult.Image);
            return Task.CompletedTask;
        }


        /// <summary>
        /// Paste the image.
        /// </summary>
        /// <returns></returns>
        private Task PasteImage()
        {
            return HandleClipboardInput();
        }


        /// <summary>
        /// Handles the clipboard input.
        /// </summary>
        /// <returns></returns>
        private Task HandleClipboardInput()
        {
            if (Clipboard.ContainsImage())
                ShowCropImageDialog(Clipboard.GetImage());
            else if (Clipboard.ContainsFileDropList())
            {
                var imageFile = Clipboard.GetFileDropList()
                    .OfType<string>()
                    .FirstOrDefault();
                ShowCropImageDialog(null, imageFile);
            }
            return Task.CompletedTask;
        }


        /// <summary>
        /// Handles the MouseLeftButtonDown event of the PaintCanvas control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.Windows.Input.MouseButtonEventArgs"/> instance containing the event data.</param>
        private async void PaintCanvas_MouseLeftButtonDown(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            await SaveCanvas();
        }


        /// <summary>
        /// Handles the MouseLeftButtonUp event of the PaintCanvas control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="MouseButtonEventArgs"/> instance containing the event data.</param>
        private async void PaintCanvas_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            await SaveCanvas();
        }

        /// <summary>
        /// Called on key down.
        /// </summary>
        /// <param name="sender">The sender.</param>
        /// <param name="e">The <see cref="KeyEventArgs"/> instance containing the event data.</param>
        private async void OnPreviewKeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.V && Keyboard.Modifiers == ModifierKeys.Control)
            {
                await HandleClipboardInput();
                e.Handled = true;
            }
            else if (e.Key == Key.C && Keyboard.Modifiers == ModifierKeys.Control)
            {
                await CopyImage();
                e.Handled = true;
            }
        }


        /// <summary>
        /// Called when mouse enters.
        /// </summary>
        /// <param name="sender">The sender.</param>
        /// <param name="e">The <see cref="MouseEventArgs"/> instance containing the event data.</param>
        private void OnMouseEnter(object sender, MouseEventArgs e)
        {
            Focus();
        }


        /// <summary>
        /// Handles the PreviewMouseMove event of the PaintCanvas control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="MouseEventArgs"/> instance containing the event data.</param>
        private async void PaintCanvas_PreviewMouseMove(object sender, MouseEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                if (DateTime.Now > _canvasLastUpdate)
                {
                    _canvasLastUpdate = DateTime.Now.AddMilliseconds(_canvasUpdatedelay);
                    await SaveCanvas();
                }
            }
        }


        /// <summary>
        /// Called when [preview drop].
        /// </summary>
        /// <param name="sender">The sender.</param>
        /// <param name="e">The <see cref="DragEventArgs"/> instance containing the event data.</param>
        private void OnPreviewDrop(object sender, DragEventArgs e)
        {
            var fileNames = (string[])e.Data.GetData(DataFormats.FileDrop);
            if (!fileNames.IsNullOrEmpty())
                ShowCropImageDialog(null, fileNames.FirstOrDefault());
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
