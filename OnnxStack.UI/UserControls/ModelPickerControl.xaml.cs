using Models;
using OnnxStack.Core.Services;
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
        private readonly IOnnxModelService _modelService;

        /// <summary>Initializes a new instance of the <see cref="ModelPickerControl" /> class.</summary>
        public ModelPickerControl()
        {
            if (!DesignerProperties.GetIsInDesignMode(this))
                _modelService = App.GetService<IOnnxModelService>();

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
            if (_modelService.IsModelLoaded(SelectedModel.ModelOptions))
                return;

            SelectedModel.IsLoading = true;
            await _modelService.LoadModel(SelectedModel.ModelOptions);
            SelectedModel.IsLoading = false;
            SelectedModel.IsLoaded = true;
        }


        /// <summary>
        /// Unloads the model.
        /// </summary>
        private async Task UnloadModel()
        {
            if (!_modelService.IsModelLoaded(SelectedModel.ModelOptions))
                return;

            SelectedModel.IsLoading = true;
            await _modelService.UnloadModel(SelectedModel.ModelOptions);
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
