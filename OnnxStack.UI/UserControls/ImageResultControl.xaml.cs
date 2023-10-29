using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace OnnxStack.UI.UserControls
{
    public partial class ImageResultControl : UserControl, INotifyPropertyChanged
    {

        /// <summary>Initializes a new instance of the <see cref="ImageResultControl" /> class.</summary>
        public ImageResultControl()
        {
            CopyImageCommand = new AsyncRelayCommand(CopyImage);
            UpdateSeedCommand = new AsyncRelayCommand<int>(UpdateSeed);
            InitializeComponent();
        }

        public AsyncRelayCommand CopyImageCommand { get; }
        public AsyncRelayCommand<int> UpdateSeedCommand { get; }

        public ImageResult Result
        {
            get { return (ImageResult)GetValue(ResultProperty); }
            set { SetValue(ResultProperty, value); }
        }

        public static readonly DependencyProperty ResultProperty =
            DependencyProperty.Register("Result", typeof(ImageResult), typeof(ImageResultControl));


        public SchedulerOptionsModel SchedulerOptions
        {
            get { return (SchedulerOptionsModel)GetValue(SchedulerOptionsProperty); }
            set { SetValue(SchedulerOptionsProperty, value); }
        }

        public static readonly DependencyProperty SchedulerOptionsProperty =
            DependencyProperty.Register("SchedulerOptions", typeof(SchedulerOptionsModel), typeof(ImageResultControl));


        public int ProgressMax
        {
            get { return (int)GetValue(ProgressMaxProperty); }
            set { SetValue(ProgressMaxProperty, value); }
        }

        public static readonly DependencyProperty ProgressMaxProperty =
            DependencyProperty.Register("ProgressMax", typeof(int), typeof(ImageResultControl));


        public int ProgressValue
        {
            get { return (int)GetValue(ProgressValueProperty); }
            set { SetValue(ProgressValueProperty, value); }
        }

        public static readonly DependencyProperty ProgressValueProperty =
            DependencyProperty.Register("ProgressValue", typeof(int), typeof(ImageResultControl));


        public bool HasResult
        {
            get { return (bool)GetValue(HasResultProperty); }
            set { SetValue(HasResultProperty, value); }
        }

        public static readonly DependencyProperty HasResultProperty =
            DependencyProperty.Register("HasResult", typeof(bool), typeof(ImageResultControl));



        private Task UpdateSeed(int previousSeed)
        {
            SchedulerOptions.Seed = previousSeed;
            return Task.CompletedTask;
        }

        private Task CopyImage()
        {
            if (Result?.Image != null)
                Clipboard.SetImage(Result.Image);

            return Task.CompletedTask;
        }

        private async void OnPreviewKeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.C && Keyboard.Modifiers == ModifierKeys.Control)
            {
                await CopyImage();
                e.Handled = true;
            }
        }

        private void OnMouseEnter(object sender, MouseEventArgs e)
        {
            Focus();
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
