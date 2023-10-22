using OnnxStack.UI.Commands;
using OnnxStack.UI.Dialogs;
using OnnxStack.UI.Models;
using OnnxStack.UI.Services;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace OnnxStack.UI.UserControls
{
    public partial class ImageInputControl : UserControl, INotifyPropertyChanged
    {
        private readonly IDialogService _dialogService;

        /// <summary>Initializes a new instance of the <see cref="ImageInputControl" /> class.</summary>
        public ImageInputControl()
        {
            if (!DesignerProperties.GetIsInDesignMode(this))
                _dialogService = App.GetService<IDialogService>();

            LoadImageCommand = new AsyncRelayCommand(LoadImage);
            ClearImageCommand = new AsyncRelayCommand(ClearImage);
            InitializeComponent();
        }

        public AsyncRelayCommand LoadImageCommand { get; }
        public AsyncRelayCommand ClearImageCommand { get; }

        public ImageInput Result
        {
            get { return (ImageInput)GetValue(ResultProperty); }
            set { SetValue(ResultProperty, value); }
        }

        public static readonly DependencyProperty ResultProperty =
            DependencyProperty.Register("Result", typeof(ImageInput), typeof(ImageInputControl));


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


        private Task ClearImage()
        {
            Result = null;
            HasResult = false;
            return Task.CompletedTask;
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
