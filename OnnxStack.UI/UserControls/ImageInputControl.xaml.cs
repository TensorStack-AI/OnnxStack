using OnnxStack.UI.Commands;
using OnnxStack.UI.Dialogs;
using OnnxStack.UI.Models;
using OnnxStack.UI.Services;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Ink;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace OnnxStack.UI.UserControls
{
    public partial class ImageInputControl : UserControl, INotifyPropertyChanged
    {
        private readonly IDialogService _dialogService;

        private int _maskDrawSize = 20;
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

            LoadImageCommand = new AsyncRelayCommand(LoadImage);
            ClearImageCommand = new AsyncRelayCommand(ClearImage);
            MaskModeCommand = new AsyncRelayCommand(MaskMode);
            SaveMaskCommand = new AsyncRelayCommand(SaveMask);
            InitializeComponent();
        }

        public AsyncRelayCommand LoadImageCommand { get; }
        public AsyncRelayCommand ClearImageCommand { get; }
        public AsyncRelayCommand MaskModeCommand { get; }
        public AsyncRelayCommand SaveMaskCommand { get; }

        public ImageInput Result
        {
            get { return (ImageInput)GetValue(ResultProperty); }
            set { SetValue(ResultProperty, value); }
        }

        public static readonly DependencyProperty ResultProperty =
            DependencyProperty.Register("Result", typeof(ImageInput), typeof(ImageInputControl));

        public ImageInput MaskResult
        {
            get { return (ImageInput)GetValue(MaskResultProperty); }
            set { SetValue(MaskResultProperty, value); }
        }

        public static readonly DependencyProperty MaskResultProperty =
            DependencyProperty.Register("MaskResult", typeof(ImageInput), typeof(ImageInputControl));

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
            var loadImageDialog = _dialogService.GetDialog<CropImageDialog>();
            loadImageDialog.Initialize(SchedulerOptions.Width, SchedulerOptions.Height);
            if (loadImageDialog.ShowDialog() == true)
            {
                Result = new ImageInput
                {
                    Image = loadImageDialog.GetImageResult(),
                    FileName = loadImageDialog.ImageFile,
                };
                HasResult = true;
            }
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
            HasMaskResult = false;
            MaskCanvas.Strokes.Clear();
            IsMaskEraserEnabled = false;
            MaskEditingMode = InkCanvasEditingMode.Ink;
            return Task.CompletedTask;
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

        #region INotifyPropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;
        public void NotifyPropertyChanged([CallerMemberName] string property = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(property));
        }
        #endregion
    }
}
