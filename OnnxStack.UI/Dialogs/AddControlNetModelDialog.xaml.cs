using Microsoft.Extensions.Logging;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
using OnnxStack.UI.Services;
using System;
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
    /// Interaction logic for AddControlNetModelDialog.xaml
    /// </summary>
    public partial class AddControlNetModelDialog : Window, INotifyPropertyChanged
    {
        private readonly ILogger<AddControlNetModelDialog> _logger;

        private readonly List<string> _invalidOptions;
        private string _modelName;
        private string _modelFile;
        private string _annotationModelFile;
        private ControlNetType _selectedControlNetType;
        private IModelFactory _modelFactory;
        private OnnxStackUIConfig _settings;
        private ControlNetModelSet _modelSetResult;

        public AddControlNetModelDialog(OnnxStackUIConfig settings, IModelFactory modelFactory, ILogger<AddControlNetModelDialog> logger)
        {
            _logger = logger;
            _settings = settings;
            _modelFactory = modelFactory;
            SaveCommand = new AsyncRelayCommand(Save, CanExecuteSave);
            CancelCommand = new AsyncRelayCommand(Cancel);
            _invalidOptions = _settings.GetModelNames();
            InitializeComponent();
            SelectedControlNetType = ControlNetType.Canny;
        }

        public AsyncRelayCommand SaveCommand { get; }
        public AsyncRelayCommand CancelCommand { get; }
        public ObservableCollection<ValidationResult> ValidationResults { get; set; } = new ObservableCollection<ValidationResult>();

        public string ModelName
        {
            get { return _modelName; }
            set { _modelName = value; _modelName?.Trim(); NotifyPropertyChanged(); CreateModelSet(); }
        }

        public string ModelFile
        {
            get { return _modelFile; }
            set
            {
                _modelFile = value;
                _modelName = string.IsNullOrEmpty(_modelFile)
                    ? string.Empty
                    : Path.GetFileNameWithoutExtension(_modelFile);

                NotifyPropertyChanged();
                NotifyPropertyChanged(nameof(ModelName));
                CreateModelSet();
            }
        }

        public string AnnotationModelFile
        {
            get { return _annotationModelFile; }
            set { _annotationModelFile = value; NotifyPropertyChanged(); CreateModelSet(); }
        }

        public ControlNetType SelectedControlNetType
        {
            get { return _selectedControlNetType; }
            set { _selectedControlNetType = value; NotifyPropertyChanged(); CreateModelSet(); }

        }

        public ControlNetModelSet ModelSetResult
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
            if (string.IsNullOrEmpty(_modelFile))
                return;

            _modelSetResult = _modelFactory.CreateControlNetModelSet(ModelName.Trim(), _selectedControlNetType, _modelFile, _annotationModelFile);

            // Validate
            ValidationResults.Add(new ValidationResult("Name", !_invalidOptions.Contains(_modelName, StringComparer.OrdinalIgnoreCase) && _modelName.Length > 2 && _modelName.Length < 50));
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
            if (string.IsNullOrEmpty(_modelFile))
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
