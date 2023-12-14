using Microsoft.Extensions.Logging;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
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
    /// Interaction logic for UpdateUpscaleModelDialog.xaml
    /// </summary>
    public partial class UpdateUpscaleModelDialog : Window, INotifyPropertyChanged
    {
        private readonly ILogger<UpdateUpscaleModelDialog> _logger;

        private List<string> _invalidOptions;
        private string _modelFile;
        private string _modelName;
        private OnnxStackUIConfig _uiSettings;

        public UpdateUpscaleModelDialog(OnnxStackUIConfig uiSettings, ILogger<UpdateUpscaleModelDialog> logger)
        {
            _logger = logger;
            _uiSettings = uiSettings;
            WindowCloseCommand = new AsyncRelayCommand(WindowClose);
            WindowRestoreCommand = new AsyncRelayCommand(WindowRestore);
            WindowMinimizeCommand = new AsyncRelayCommand(WindowMinimize);
            WindowMaximizeCommand = new AsyncRelayCommand(WindowMaximize);
            SaveCommand = new AsyncRelayCommand(Save, CanExecuteSave);
            CancelCommand = new AsyncRelayCommand(Cancel, CanExecuteCancel);
            InitializeComponent();
        }
        public AsyncRelayCommand WindowMinimizeCommand { get; }
        public AsyncRelayCommand WindowRestoreCommand { get; }
        public AsyncRelayCommand WindowMaximizeCommand { get; }
        public AsyncRelayCommand WindowCloseCommand { get; }
        public AsyncRelayCommand SaveCommand { get; }
        public AsyncRelayCommand CancelCommand { get; }

        public ObservableCollection<ValidationResult> ValidationResults { get; set; } = new ObservableCollection<ValidationResult>();


        public string ModelName
        {
            get { return _modelName; }
            set { _modelName = value; NotifyPropertyChanged(); CreateModelSet(); }
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

        private bool _isNameInvalid;

        public bool IsNameInvalid
        {
            get { return _isNameInvalid; }
            set { _isNameInvalid = value; NotifyPropertyChanged(); }
        }


        private UpscaleModelSet _modelSet;

        public UpscaleModelSet ModelSet
        {
            get { return _modelSet; }
            set { _modelSet = value; NotifyPropertyChanged(); }
        }



        private void CreateModelSet()
        {
            ModelSet = null;
            IsNameInvalid = false;
            ValidationResults.Clear();
            if (string.IsNullOrEmpty(_modelFile))
                return;

            ModelSet = new UpscaleModelSet
            {
                Name = ModelName.Trim(),
                Channels = 3,
                ScaleFactor = 4,
                SampleSize = 512,

                DeviceId = _uiSettings.DefaultDeviceId,
                ExecutionMode = _uiSettings.DefaultExecutionMode,
                ExecutionProvider = _uiSettings.DefaultExecutionProvider,
                InterOpNumThreads = _uiSettings.DefaultInterOpNumThreads,
                IntraOpNumThreads = _uiSettings.DefaultIntraOpNumThreads,
                IsEnabled = true,
                ModelConfigurations = new List<OnnxModelConfig>
                {
                    new OnnxModelConfig { Type = OnnxModelType.Unet, OnnxModelPath = _modelFile }
                }
            };

            // Validate
            IsNameInvalid = !InvalidOptions.IsNullOrEmpty() && InvalidOptions.Contains(_modelName);
            foreach (var validationResult in ModelSet.ModelConfigurations.Select(x => new ValidationResult(x.Type, File.Exists(x.OnnxModelPath))))
            {
                ValidationResults.Add(validationResult);
            }
        }


        public List<string> InvalidOptions
        {
            get { return _invalidOptions; }
            set { _invalidOptions = value; NotifyPropertyChanged(); }
        }


        public bool ShowDialog(UpscaleModelSet modelSet, List<string> invalidOptions = null)
        {
            ModelSet = modelSet with { };
            InvalidOptions = invalidOptions;
            return base.ShowDialog() ?? false;
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
            if (string.IsNullOrEmpty(_modelName) || IsNameInvalid)
                return false;
            if (_modelSet is null)
                return false;

            var result = _modelName.Trim();
            if (!InvalidOptions.IsNullOrEmpty() && InvalidOptions.Contains(result))
                return false;

            return (result.Length > 2 && result.Length <= 50)
            && (ValidationResults.Count > 0 && ValidationResults.All(x => x.IsValid));
        }

        private Task Cancel()
        {
            ModelSet = null;
            DialogResult = false;
            return Task.CompletedTask;
        }

        private bool CanExecuteCancel()
        {
            return true;
        }

        #region BaseWindow

        private Task WindowClose()
        {
            Close();
            return Task.CompletedTask;
        }

        private Task WindowRestore()
        {
            if (WindowState == WindowState.Maximized)
                WindowState = WindowState.Normal;
            else
                WindowState = WindowState.Maximized;
            return Task.CompletedTask;
        }

        private Task WindowMinimize()
        {
            WindowState = WindowState.Minimized;
            return Task.CompletedTask;
        }

        private Task WindowMaximize()
        {
            WindowState = WindowState.Maximized;
            return Task.CompletedTask;
        }

        private void OnContentRendered(object sender, EventArgs e)
        {
            InvalidateVisual();
        }
        #endregion

        #region INotifyPropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;
        public void NotifyPropertyChanged([CallerMemberName] string property = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(property));
        }
        #endregion
    }

}
