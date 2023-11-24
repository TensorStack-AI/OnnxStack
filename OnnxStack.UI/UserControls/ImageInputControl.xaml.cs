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
    public partial class ImageInputControl : UserControl, INotifyPropertyChanged
    {
        private readonly IDialogService _dialogService;

        private int _maskDrawSize;
        private bool _hasMaskChanged;
        private bool _isMaskEraserEnabled;
        private DrawingAttributes _maskAttributes;
        private InkCanvasEditingMode _maskEditingMode = InkCanvasEditingMode.Ink;

        /// <summary>
        /// Initializes a new instance of the <see cref="ImageInputControl" /> class.
        /// </summary>
        public ImageInputControl()
        {
            if (!DesignerProperties.GetIsInDesignMode(this))
                _dialogService = App.GetService<IDialogService>();

            MaskDrawSize = 20;
            LoadImageCommand = new AsyncRelayCommand(LoadImage);
            ClearImageCommand = new AsyncRelayCommand(ClearImage);
            MaskModeCommand = new AsyncRelayCommand(MaskMode);
            CopyImageCommand = new AsyncRelayCommand(CopyImage);
            PasteImageCommand = new AsyncRelayCommand(PasteImage);
            InitializeComponent();
        }

        public AsyncRelayCommand LoadImageCommand { get; }
        public AsyncRelayCommand ClearImageCommand { get; }
        public AsyncRelayCommand MaskModeCommand { get; }
        public AsyncRelayCommand CopyImageCommand { get; }
        public AsyncRelayCommand PasteImageCommand { get; }
        public ImageInput Result
        {
            get { return (ImageInput)GetValue(ResultProperty); }
            set { SetValue(ResultProperty, value); }
        }

        public static readonly DependencyProperty ResultProperty =
            DependencyProperty.Register("Result", typeof(ImageInput), typeof(ImageInputControl), new PropertyMetadata((s, e) =>
            {
                if (s is ImageInputControl control)
                    control.SaveMask();
            }));

        public ImageInput MaskResult
        {
            get { return (ImageInput)GetValue(MaskResultProperty); }
            set { SetValue(MaskResultProperty, value); }
        }

        public static readonly DependencyProperty MaskResultProperty =
            DependencyProperty.Register("MaskResult", typeof(ImageInput), typeof(ImageInputControl), new PropertyMetadata((s, e) =>
            {
                if (e.NewValue is null && s is ImageInputControl control)
                    control.ClearMask();
            }));

        public SchedulerOptionsModel SchedulerOptions
        {
            get { return (SchedulerOptionsModel)GetValue(SchedulerOptionsProperty); }
            set { SetValue(SchedulerOptionsProperty, value); }
        }

        public static readonly DependencyProperty SchedulerOptionsProperty =
            DependencyProperty.Register("SchedulerOptions", typeof(SchedulerOptionsModel), typeof(ImageInputControl));

        public bool HasResult
        {
            get { return (bool)GetValue(HasResultProperty); }
            set { SetValue(HasResultProperty, value); }
        }

        public static readonly DependencyProperty HasResultProperty =
            DependencyProperty.Register("HasResult", typeof(bool), typeof(ImageInputControl));

        public bool HasMaskResult
        {
            get { return (bool)GetValue(HasMaskResultProperty); }
            set { SetValue(HasMaskResultProperty, value); }
        }

        public static readonly DependencyProperty HasMaskResultProperty =
            DependencyProperty.Register("HasMaskResult", typeof(bool), typeof(ImageInputControl));

        public bool IsMaskEnabled
        {
            get { return (bool)GetValue(IsMaskEnabledProperty); }
            set { SetValue(IsMaskEnabledProperty, value); }
        }

        // Using a DependencyProperty as the backing store for IsMaskEnabled.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty IsMaskEnabledProperty =
            DependencyProperty.Register("IsMaskEnabled", typeof(bool), typeof(ImageInputControl));


        public InkCanvasEditingMode MaskEditingMode
        {
            get { return _maskEditingMode; }
            set { _maskEditingMode = value; NotifyPropertyChanged(); }
        }

        public DrawingAttributes MaskAttributes
        {
            get { return _maskAttributes; }
            set { _maskAttributes = value; NotifyPropertyChanged(); }
        }

        public bool IsMaskEraserEnabled
        {
            get { return _isMaskEraserEnabled; }
            set { _isMaskEraserEnabled = value; NotifyPropertyChanged(); }
        }

        public bool HasMaskChanged
        {
            get { return _hasMaskChanged; }
            set { _hasMaskChanged = value; NotifyPropertyChanged(); }
        }

        public int MaskDrawSize
        {
            get { return _maskDrawSize; }
            set
            {
                _maskDrawSize = value;
                NotifyPropertyChanged();
                UpdateMaskAttributes();
            }
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
            Result = null;
            MaskResult = null;
            HasResult = false;
            ClearMask();
            return Task.CompletedTask;
        }


        /// <summary>
        /// Clears the mask.
        /// </summary>
        private void ClearMask()
        {
            HasMaskResult = false;
            HasMaskChanged = false;
            MaskCanvas.Strokes.Clear();
            IsMaskEraserEnabled = false;
            MaskEditingMode = InkCanvasEditingMode.Ink;
        }


        /// <summary>
        /// Saves the mask.
        /// </summary>
        /// <returns></returns>
        private Task SaveMask()
        {
            MaskResult = new ImageInput
            {
                Image = CreateMaskImage(),
                FileName = "OnnxStack Generated Mask",
            };
            HasMaskResult = true;
            HasMaskChanged = false;
            return Task.CompletedTask;
        }


        /// <summary>
        /// Change Masks mode.
        /// </summary>
        /// <returns></returns>
        private Task MaskMode()
        {
            if (_isMaskEraserEnabled)
            {
                IsMaskEraserEnabled = false;
                MaskEditingMode = InkCanvasEditingMode.Ink;
            }
            else
            {
                IsMaskEraserEnabled = true;
                MaskEditingMode = InkCanvasEditingMode.EraseByPoint;
            }
            return Task.CompletedTask;
        }


        /// <summary>
        /// Updates the mask attributes.
        /// </summary>
        private void UpdateMaskAttributes()
        {
            MaskAttributes = new DrawingAttributes
            {
                Color = Colors.Black,
                Height = _maskDrawSize,
                Width = _maskDrawSize,
            };
        }


        /// <summary>
        /// Creates the mask image.
        /// </summary>
        /// <returns></returns>
        public BitmapSource CreateMaskImage()
        {
            if (MaskCanvas.ActualWidth == 0)
                return CreateEmptyMaskImage();

            // Create a RenderTargetBitmap to render the Canvas content.
            var renderBitmap = new RenderTargetBitmap((int)MaskCanvas.ActualWidth, (int)MaskCanvas.ActualHeight, 96, 96, PixelFormats.Pbgra32);

            // Make a drawing visual to render.
            var visual = new DrawingVisual();
            using (DrawingContext context = visual.RenderOpen())
            {
                VisualBrush brush = new VisualBrush(MaskCanvas);
                context.DrawRectangle(brush, null, new Rect(new Point(0, 0), new Point(MaskCanvas.ActualWidth, MaskCanvas.ActualHeight)));
            }
            renderBitmap.Render(visual);
            return renderBitmap;
        }


        public BitmapSource CreateEmptyMaskImage()
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
                await ClearImage();
                Result = new ImageInput
                {
                    Image = loadImageDialog.GetImageResult(),
                    FileName = loadImageDialog.ImageFile,
                };
                HasResult = true;
                await SaveMask();
            }
        }


        /// <summary>
        /// Copies the image.
        /// </summary>
        /// <returns></returns>
        private Task CopyImage()
        {
            if (Result?.Image != null)
                Clipboard.SetImage(Result.Image);
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
        /// Handles the MouseLeftButtonDown event of the MaskCanvas control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.Windows.Input.MouseButtonEventArgs"/> instance containing the event data.</param>
        private void MaskCanvas_MouseLeftButtonDown(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            HasMaskResult = false;
            HasMaskChanged = true;
        }


        /// <summary>
        /// Handles the MouseLeftButtonUp event of the MaskCanvas control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="MouseButtonEventArgs"/> instance containing the event data.</param>
        private async void MaskCanvas_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            await SaveMask();
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
