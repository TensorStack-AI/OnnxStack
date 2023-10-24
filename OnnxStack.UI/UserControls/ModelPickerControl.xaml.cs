using Models;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.UI.Commands;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace OnnxStack.UI.UserControls
{
    /// <summary>
    /// Interaction logic for Parameters.xaml
    /// </summary>
    public partial class ModelPickerControl : UserControl, INotifyPropertyChanged
    {
        private readonly IStableDiffusionService _stableDiffusionService;

        /// <summary>Initializes a new instance of the <see cref="ModelPickerControl" /> class.</summary>
        public ModelPickerControl()
        {
            if (!DesignerProperties.GetIsInDesignMode(this))
                _stableDiffusionService = App.GetService<IStableDiffusionService>();

            LoadModelCommand = new AsyncRelayCommand(LoadModel);
            UnloadModelCommand = new AsyncRelayCommand(UnloadModel);
            InitializeComponent();
        }

        public AsyncRelayCommand LoadModelCommand { get; set; }
        public AsyncRelayCommand UnloadModelCommand { get; set; }

        public ModelOptionsModel SelectedModel
        {
            get { return (ModelOptionsModel)GetValue(SelectedModelProperty); }
            set { SetValue(SelectedModelProperty, value); }
        }


        public static readonly DependencyProperty SelectedModelProperty =
            DependencyProperty.Register("SelectedModel", typeof(ModelOptionsModel), typeof(ModelPickerControl));


        /// <summary>
        /// Loads the model.
        /// </summary>
        private async Task LoadModel()
        {
            if (_stableDiffusionService.IsModelLoaded(SelectedModel.ModelOptions))
                return;

            SelectedModel.IsLoading = true;
            var result = await _stableDiffusionService.LoadModel(SelectedModel.ModelOptions);
            SelectedModel.IsLoading = false;
            SelectedModel.IsLoaded = result;
        }


        /// <summary>
        /// Unloads the model.
        /// </summary>
        private async Task UnloadModel()
        {
            if (!_stableDiffusionService.IsModelLoaded(SelectedModel.ModelOptions))
                return;

            SelectedModel.IsLoading = true;
            await _stableDiffusionService.UnloadModel(SelectedModel.ModelOptions);
            SelectedModel.IsLoading = false;
            SelectedModel.IsLoaded = false;
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
