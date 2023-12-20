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
    /// Interaction logic for UpdateModelDialog.xaml
    /// </summary>
    public partial class UpdateModelDialog : Window, INotifyPropertyChanged
    {
        private List<string> _invalidOptions;
        private OnnxStackUIConfig _uiSettings;
        private UpdateModelSetViewModel _updateModelSet;
        private StableDiffusionModelSet _modelSetResult;
        private string _validationError;

        public UpdateModelDialog(OnnxStackUIConfig uiSettings)
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

        public UpdateModelSetViewModel UpdateModelSet
        {
            get { return _updateModelSet; }
            set { _updateModelSet = value; NotifyPropertyChanged(); }
        }

        public string ValidationError
        {
            get { return _validationError; }
            set { _validationError = value; NotifyPropertyChanged(); }
        }

        public StableDiffusionModelSet ModelSetResult
        {
            get { return _modelSetResult; }
        }


        public bool ShowDialog(StableDiffusionModelSet modelSet)
        {
            _invalidOptions.Remove(modelSet.Name);
            UpdateModelSet = UpdateModelSetViewModel.FromModelSet(modelSet);
            return ShowDialog() ?? false;
        }


        private Task Save()
        {
            _modelSetResult = UpdateModelSetViewModel.ToModelSet(_updateModelSet);
            if (_invalidOptions.Contains(_modelSetResult.Name))
            {
                ValidationError = $"Model with name '{_modelSetResult.Name}' already exists";
                return Task.CompletedTask;
            }

            foreach (var modelFile in _modelSetResult.ModelConfigurations)
            {
                if (!File.Exists(modelFile.OnnxModelPath))
                {
                    ValidationError = $"'{modelFile.Type}' model file not found";
                    return Task.CompletedTask;
                }
            }

            DialogResult = true;
            return Task.CompletedTask;
        }


        private bool CanExecuteSave()
        {
            return true;
        }


        private Task Cancel()
        {
            _modelSetResult = null;
            UpdateModelSet = null;
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
