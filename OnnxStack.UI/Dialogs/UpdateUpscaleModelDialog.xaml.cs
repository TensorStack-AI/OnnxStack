using OnnxStack.ImageUpscaler.Common;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
using System.Collections.Generic;
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
        private List<string> _invalidOptions;
        private OnnxStackUIConfig _uiSettings;
        private UpscaleModelSet _modelSetResult;
        private UpdateUpscaleModelSetViewModel _updateModelSet;
        private string _validationError;

        public UpdateUpscaleModelDialog(OnnxStackUIConfig uiSettings)
        {
            _uiSettings = uiSettings;
            SaveCommand = new AsyncRelayCommand(Save, CanExecuteSave);
            CancelCommand = new AsyncRelayCommand(Cancel, CanExecuteCancel);
            _invalidOptions = uiSettings.GetModelNames();
            InitializeComponent();
        }

        public OnnxStackUIConfig UISettings => _uiSettings;
        public AsyncRelayCommand SaveCommand { get; }
        public AsyncRelayCommand CancelCommand { get; }

        public UpdateUpscaleModelSetViewModel UpdateModelSet
        {
            get { return _updateModelSet; }
            set { _updateModelSet = value; NotifyPropertyChanged(); }
        }

        public string ValidationError
        {
            get { return _validationError; }
            set { _validationError = value; NotifyPropertyChanged(); }
        }

        public UpscaleModelSet ModelSetResult
        {
            get { return _modelSetResult; }
        }


        public bool ShowDialog(UpscaleModelSet modelSet)
        {
            _invalidOptions.Remove(modelSet.Name);
            UpdateModelSet = UpdateUpscaleModelSetViewModel.FromModelSet(modelSet);
            return base.ShowDialog() ?? false;
        }

        private bool Validate()
        {
            if (_updateModelSet == null)
                return false;

            _modelSetResult = UpdateUpscaleModelSetViewModel.ToModelSet(_updateModelSet);
            if (_modelSetResult == null)
                return false;

            if (_invalidOptions.Contains(_modelSetResult.Name))
            {
                ValidationError = $"Model with name '{_modelSetResult.Name}' already exists";
                return false;
            }

            if (!File.Exists(_modelSetResult.UpscaleModelConfig.OnnxModelPath))
            {
                ValidationError = $"Upscale model file not found";
                return false;
            }

            ValidationError = null;
            return true;
        }

        private Task Save()
        {

            if (!Validate())
                return Task.CompletedTask;

            DialogResult = true;
            return Task.CompletedTask;
        }


        private bool CanExecuteSave()
        {
            return Validate();
        }


        private Task Cancel()
        {
            _modelSetResult = null;
            DialogResult = false;
            return Task.CompletedTask;
        }


        private bool CanExecuteCancel()
        {
            return true;
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
