using Microsoft.Extensions.Logging;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
using OnnxStack.UI.Views;
using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;

namespace OnnxStack.UI.Dialogs
{
    /// <summary>
    /// Interaction logic for UpdateModelMetadataDialog.xaml
    /// </summary>
    public partial class UpdateModelMetadataDialog : Window, INotifyPropertyChanged
    {
        private readonly ILogger<UpdateModelMetadataDialog> _logger;

        private OnnxStackUIConfig _uiSettings;
        private string _validationError;

        public UpdateModelMetadataDialog(OnnxStackUIConfig uiSettings, ILogger<UpdateModelMetadataDialog> logger)
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


        public string ValidationError
        {
            get { return _validationError; }
            set { _validationError = value; NotifyPropertyChanged(); }
        }

        private string _website;
        public string Website
        {
            get { return _website; }
            set { _website = value; NotifyPropertyChanged(); }
        }

        private string _author;

        public string Author
        {
            get { return _author; }
            set { _author = value; NotifyPropertyChanged(); }
        }

        private string _description;

        public string Description
        {
            get { return _description; }
            set { _description = value; NotifyPropertyChanged(); }
        }


        private string _iconImage;

        public string IconImage
        {
            get { return _iconImage; }
            set { _iconImage = value; NotifyPropertyChanged(); }
        }

        private string _repository;

        public string Repository
        {
            get { return _repository; }
            set { _repository = value; NotifyPropertyChanged(); }
        }

        private string _repositoryBranch;

        public string RepositoryBranch
        {
            get { return _repositoryBranch; }
            set { _repositoryBranch = value; NotifyPropertyChanged(); }
        }

        public ObservableCollection<string> PreviewImages { get; set; } = new ObservableCollection<string>();

        public ObservableCollection<string> RepositoryFiles { get; set; } = new ObservableCollection<string>();

        private ModelTemplateViewModel _modelTemplate;
        public bool ShowDialog(ModelTemplateViewModel modelTemplate)
        {
            _modelTemplate = modelTemplate;

            Website = _modelTemplate.Website;
            Author = _modelTemplate.Author;
            Description = _modelTemplate.Description;
            IconImage = _modelTemplate.ImageIcon;
            Repository = _modelTemplate.Repository;
            RepositoryBranch = _modelTemplate.RepositoryBranch;
           
            for (int i = 0; i < 4; i++)
            {
                PreviewImages.Add(_modelTemplate.PreviewImages?.ElementAtOrDefault(i));
            }
            for (int i = 0; i < 12; i++)
            {
                RepositoryFiles.Add(_modelTemplate.RepositoryFiles?.ElementAtOrDefault(i));
            }

            return ShowDialog() ?? false;
        }

        private Task Save()
        {
            // validate links;
            _modelTemplate.Website = Website;
            _modelTemplate.Author = Author;
            _modelTemplate.ImageIcon = IconImage;
            _modelTemplate.Description = Description;
            _modelTemplate.Repository = Repository;
            _modelTemplate.RepositoryBranch = RepositoryBranch;
            _modelTemplate.PreviewImages = PreviewImages.Where(x => !string.IsNullOrEmpty(x)).ToList();
            _modelTemplate.RepositoryFiles = RepositoryFiles.Where(x => !string.IsNullOrEmpty(x)).ToList();

            DialogResult = true;
            return Task.CompletedTask;
        }

        private bool CanExecuteSave()
        {
            return true;
        }

        private Task Cancel()
        {
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
