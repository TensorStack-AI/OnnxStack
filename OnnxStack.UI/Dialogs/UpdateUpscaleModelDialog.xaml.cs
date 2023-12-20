using OnnxStack.StableDiffusion.Config;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
using System;
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
            WindowCloseCommand = new AsyncRelayCommand(WindowClose);
            WindowRestoreCommand = new AsyncRelayCommand(WindowRestore);
            WindowMinimizeCommand = new AsyncRelayCommand(WindowMinimize);
            WindowMaximizeCommand = new AsyncRelayCommand(WindowMaximize);
            SaveCommand = new AsyncRelayCommand(Save, CanExecuteSave);
            CancelCommand = new AsyncRelayCommand(Cancel, CanExecuteCancel);
            _invalidOptions = _uiSettings.Templates
               .Where(x => x.IsUserTemplate)
               .Select(x => x.Name)
               .ToList();
            InitializeComponent();
        }
        public AsyncRelayCommand WindowMinimizeCommand { get; }
        public AsyncRelayCommand WindowRestoreCommand { get; }
        public AsyncRelayCommand WindowMaximizeCommand { get; }
        public AsyncRelayCommand WindowCloseCommand { get; }
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

            foreach (var modelFile in _modelSetResult.ModelConfigurations)
            {
                if (!File.Exists(modelFile.OnnxModelPath))
                {
                    ValidationError = $"'{modelFile.Type}' model file not found";
                    return false;
                }
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
