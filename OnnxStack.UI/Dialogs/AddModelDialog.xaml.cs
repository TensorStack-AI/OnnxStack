using Microsoft.Extensions.Logging;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
using OnnxStack.UI.Services;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;

namespace OnnxStack.UI.Dialogs
{
    /// <summary>
    /// Interaction logic for AddModelDialog.xaml
    /// </summary>
    public partial class AddModelDialog : Window, INotifyPropertyChanged
    {
        private readonly ILogger<AddModelDialog> _logger;

        private List<string> _invalidOptions;
        private string _modelFolder;
        private string _modelName;
        private IModelFactory _modelFactory;
        private OnnxStackUIConfig _settings;
        private StableDiffusionModelTemplate _modelTemplate;
        private StableDiffusionModelSet _modelSetResult;

        public AddModelDialog(OnnxStackUIConfig settings, IModelFactory modelFactory, ILogger<AddModelDialog> logger)
        {
            _logger = logger;
            _settings = settings;
            _modelFactory = modelFactory;
            SaveCommand = new AsyncRelayCommand(Save, CanExecuteSave);
            CancelCommand = new AsyncRelayCommand(Cancel);
            ModelTemplates = GetModelTemplates().ToList();
            InvalidOptions = _settings.StableDiffusionModelSets.Select(x => x.Name.ToLower()).ToList();
            InitializeComponent();
        }

        private IEnumerable<StableDiffusionModelTemplate> GetModelTemplates()
        {
            yield return new StableDiffusionModelTemplate("StableDiffusion", DiffuserPipelineType.StableDiffusion, ModelType.Base, 512, DiffuserType.TextToImage, DiffuserType.ImageToImage, DiffuserType.ImageInpaintLegacy);
            yield return new StableDiffusionModelTemplate("SDXL", DiffuserPipelineType.StableDiffusionXL, ModelType.Base, 1024, DiffuserType.TextToImage, DiffuserType.ImageToImage, DiffuserType.ImageInpaintLegacy);
        }

        public AsyncRelayCommand SaveCommand { get; }
        public AsyncRelayCommand CancelCommand { get; }
        public ObservableCollection<ValidationResult> ValidationResults { get; set; } = new ObservableCollection<ValidationResult>();
        public List<StableDiffusionModelTemplate> ModelTemplates { get; set; }

        public StableDiffusionModelTemplate ModelTemplate
        {
            get { return _modelTemplate; }
            set { _modelTemplate = value; NotifyPropertyChanged(); CreateModelSet(); }
        }
        public List<string> InvalidOptions
        {
            get { return _invalidOptions; }
            set { _invalidOptions = value; NotifyPropertyChanged(); }
        }

        public string ModelName
        {
            get { return _modelName; }
            set { _modelName = value; _modelName?.Trim(); NotifyPropertyChanged(); CreateModelSet(); }
        }

        public string ModelFolder
        {
            get { return _modelFolder; }
            set
            {
                _modelFolder = value;
                if (_modelTemplate is not null)
                    _modelName = string.IsNullOrEmpty(_modelFolder)
                        ? string.Empty
                        : Path.GetFileName(_modelFolder);

                NotifyPropertyChanged();
                NotifyPropertyChanged(nameof(ModelName));
                CreateModelSet();
            }
        }

        public StableDiffusionModelSet ModelSetResult
        {
            get { return _modelSetResult; }
        }




        public new bool ShowDialog()
        {
            return base.ShowDialog() ?? false;
        }


        private void CreateModelSet()
        {
            _modelSetResult = null;
            ValidationResults.Clear();
            if (string.IsNullOrEmpty(_modelFolder))
                return;

            _modelSetResult = _modelFactory.CreateStableDiffusionModelSet(ModelName.Trim(), ModelFolder, _modelTemplate);

            // Validate
            ValidationResults.Add(new ValidationResult("Name", !InvalidOptions.Contains(_modelName.ToLower()) && _modelName.Length > 2 && _modelName.Length < 50));
            foreach (var validationResult in _modelSetResult.ModelConfigurations.Select(x => new ValidationResult(x.Type.ToString(), File.Exists(x.OnnxModelPath))))
            {
                ValidationResults.Add(validationResult);
            }
        }


        private Task Save()
        {
            DialogResult = true;
            return Task.CompletedTask;
        }


        private bool CanExecuteSave()
        {
            if (string.IsNullOrEmpty(_modelFolder))
                return false;
            if (_modelSetResult is null)
                return false;

            return ValidationResults.Count > 0 && ValidationResults.All(x => x.IsValid);
        }


        private Task Cancel()
        {
            _modelSetResult = null;
            DialogResult = false;
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
