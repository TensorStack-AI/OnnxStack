using Microsoft.Extensions.Logging;
using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Dialogs;
using OnnxStack.UI.Models;
using OnnxStack.UI.Services;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Dynamic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;

namespace OnnxStack.UI.Views
{
    /// <summary>
    /// Interaction logic for ModelSettingsView.xaml
    /// </summary>
    public partial class ModelSettingsView : UserControl, INavigatable, INotifyPropertyChanged
    {
        private readonly IModelFactory _modelFactory;
        private readonly IDialogService _dialogService;
        private readonly ILogger<ModelSettingsView> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="ModelSettingsView"/> class.
        /// </summary>
        public ModelSettingsView()
        {
            if (!DesignerProperties.GetIsInDesignMode(this))
            {
                _modelFactory = App.GetService<IModelFactory>();
                _dialogService = App.GetService<IDialogService>();
                _logger = App.GetService<ILogger<ModelSettingsView>>();
            }

            AddUpscaleModelCommand = new AsyncRelayCommand(AddUpscaleModel);
            RemoveUpscaleModelCommand = new AsyncRelayCommand(RemoveUpscaleModel, () => SelectedUpscaleModel is not null);
            UpdateUpscaleModelCommand = new AsyncRelayCommand(UpdateUpscaleModel, () => SelectedUpscaleModel is not null);
            AddStableDiffusionModelCommand = new AsyncRelayCommand(AddStableDiffusionModel);
            RemoveStableDiffusionModelCommand = new AsyncRelayCommand(RemoveStableDiffusionModel, () => SelectedStableDiffusionModel is not null);
            UpdateStableDiffusionModelCommand = new AsyncRelayCommand(UpdateStableDiffusionModel, () => SelectedStableDiffusionModel is not null);

            RenameStableDiffusionModelCommand = new AsyncRelayCommand(RenameStableDiffusionModel);
            UninstallStableDiffusionModelCommand = new AsyncRelayCommand(UninstallStableDiffusionModel);
            InstallLocalStableDiffusionModelCommand = new AsyncRelayCommand(InstallLocalStableDiffusionModel);
            InitializeComponent();
        }



        public AsyncRelayCommand AddUpscaleModelCommand { get; }
        public AsyncRelayCommand RemoveUpscaleModelCommand { get; }
        public AsyncRelayCommand UpdateUpscaleModelCommand { get; }
        public AsyncRelayCommand AddStableDiffusionModelCommand { get; }
        public AsyncRelayCommand RemoveStableDiffusionModelCommand { get; }
        public AsyncRelayCommand UpdateStableDiffusionModelCommand { get; }

        public AsyncRelayCommand RenameStableDiffusionModelCommand { get; }
        public AsyncRelayCommand UninstallStableDiffusionModelCommand { get; }
        public AsyncRelayCommand InstallLocalStableDiffusionModelCommand { get; }

        public OnnxStackUIConfig UISettings
        {
            get { return (OnnxStackUIConfig)GetValue(UISettingsProperty); }
            set { SetValue(UISettingsProperty, value); }
        }
        public static readonly DependencyProperty UISettingsProperty =
            DependencyProperty.Register("UISettings", typeof(OnnxStackUIConfig), typeof(ModelSettingsView), new PropertyMetadata(OnUISettingsChanged));

        private static void OnUISettingsChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is ModelSettingsView control && e.NewValue is OnnxStackUIConfig settings)
            {
                control.InitializeTemplates();
            }
        }



        public Task NavigateAsync(ImageResult imageResult)
        {
            throw new NotImplementedException();
        }

        private UpscaleModelSetViewModel _selectedUpscaleModel;

        public UpscaleModelSetViewModel SelectedUpscaleModel
        {
            get { return _selectedUpscaleModel; }
            set { _selectedUpscaleModel = value; NotifyPropertyChanged(); }
        }


        private StableDiffusionModelSetViewModel _selectedStableDiffusionModel;

        public StableDiffusionModelSetViewModel SelectedStableDiffusionModel
        {
            get { return _selectedStableDiffusionModel; }
            set { _selectedStableDiffusionModel = value; NotifyPropertyChanged(); }
        }

        private async Task AddUpscaleModel()
        {
            var invalidNames = UISettings.UpscaleModelSets.Select(x => x.ModelSet.Name).ToList();
            var addModelDialog = _dialogService.GetDialog<AddUpscaleModelDialog>();
            if (addModelDialog.ShowDialog(invalidNames))
            {
                UISettings.UpscaleModelSets.Add(new UpscaleModelSetViewModel
                {
                    Name = addModelDialog.ModelName,
                    ModelSet = addModelDialog.ModelSet
                });

                await SaveConfigurationFile();
            }
        }

        private async Task RemoveUpscaleModel()
        {
            UISettings.UpscaleModelSets.Remove(SelectedUpscaleModel);
            await SaveConfigurationFile();
            SelectedUpscaleModel = UISettings.UpscaleModelSets.FirstOrDefault();
        }

        private async Task UpdateUpscaleModel()
        {
            var invalidNames = UISettings.UpscaleModelSets.Select(x => x.ModelSet.Name).ToList();
            var addModelDialog = _dialogService.GetDialog<UpdateUpscaleModelDialog>();
            if (addModelDialog.ShowDialog(SelectedUpscaleModel.ModelSet, invalidNames))
            {
                SelectedUpscaleModel.Name = addModelDialog.ModelName;
                SelectedUpscaleModel.ModelSet = addModelDialog.ModelSet;
                await SaveConfigurationFile();
            }
        }



        private async Task AddStableDiffusionModel()
        {
            var invalidNames = UISettings.StableDiffusionModelSets.Select(x => x.ModelSet.Name).ToList();
            var addModelDialog = _dialogService.GetDialog<AddModelDialog>();
            if (addModelDialog.ShowDialog(invalidNames))
            {
                UISettings.StableDiffusionModelSets.Add(new StableDiffusionModelSetViewModel
                {
                    Name = addModelDialog.ModelName,
                    ModelSet = addModelDialog.ModelSet
                });

                await SaveConfigurationFile();
            }
        }

        private async Task RemoveStableDiffusionModel()
        {
            UISettings.StableDiffusionModelSets.Remove(SelectedStableDiffusionModel);
            await SaveConfigurationFile();
            SelectedStableDiffusionModel = UISettings.StableDiffusionModelSets.FirstOrDefault();
        }


        private async Task UpdateStableDiffusionModel()
        {
            var invalidNames = UISettings.StableDiffusionModelSets.Select(x => x.ModelSet.Name).ToList();
            var addModelDialog = _dialogService.GetDialog<UpdateModelDialog>();
            if (addModelDialog.ShowDialog(SelectedStableDiffusionModel.ModelSet, invalidNames))
            {
                SelectedStableDiffusionModel.Name = addModelDialog.ModelName;
                SelectedStableDiffusionModel.ModelSet = addModelDialog.ModelSet;
                await SaveConfigurationFile();
            }
        }




        private Task<bool> SaveConfigurationFile()
        {
            try
            {
                ConfigManager.SaveConfiguration(UISettings);
                return Task.FromResult(true);
            }
            catch (Exception ex)
            {
                _logger.LogError($"Error saving configuration file, {ex.Message}");
                return Task.FromResult(false);
            }
        }


        private Task Save()
        {
            try
            {
                ConfigManager.SaveConfiguration(UISettings);
            }
            catch (Exception ex)
            {
                _logger.LogError($"Error saving configuration file, {ex.Message}");
            }
            return Task.CompletedTask;
        }

        #region INotifyPropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;

        public void NotifyPropertyChanged([CallerMemberName] string property = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(property));
        }
        #endregion




        private void InitializeTemplates()
        {
            if (ModelTemplateCollectionView != null)
                ModelTemplateCollectionView.CollectionChanged -= ModelTemplateCollectionView_CollectionChanged;

            foreach (var item in UISettings.Templates)
            {
                item.IsInstalled = UISettings.StableDiffusionModelSets.Any(x => x.Name == item.Name);
            }

            ModelTemplateCollectionView = new ListCollectionView(UISettings.Templates);
            ModelTemplateSort();
            ModelTemplateCollectionView.Filter = ModelTemplateFilter;
            ModelTemplateFilterAuthors = UISettings.Templates
                .Where(x => !string.IsNullOrEmpty(x.Author))
                .Select(x => x.Author)
                .Distinct()
                .OrderBy(x => x)
                .ToList();
            ModelTemplateCollectionView.MoveCurrentToFirst();
            SelectedModelTemplate = (ModelTemplateViewModel)ModelTemplateCollectionView.CurrentItem;
            ModelTemplateCollectionView.CollectionChanged += ModelTemplateCollectionView_CollectionChanged;
        }

        private void ModelTemplateCollectionView_CollectionChanged(object sender, System.Collections.Specialized.NotifyCollectionChangedEventArgs e)
        {
            if (!ModelTemplateCollectionView.Contains(SelectedModelTemplate))
            {
                // If the selected item no longer exists(filtered) select first
                ModelTemplateCollectionView.MoveCurrentToFirst();
                SelectedModelTemplate = (ModelTemplateViewModel)ModelTemplateCollectionView.CurrentItem;
            }
        }

        private void ModelTemplateRefresh()
        {
            if (ModelTemplateCollectionView is null)
                return;
            ModelTemplateCollectionView.Refresh();
        }

        private bool ModelTemplateFilter(object obj)
        {
            if (obj is not ModelTemplateViewModel template)
                return false;

            return (string.IsNullOrEmpty(_modelTemplateFilterText) || template.Name.Contains(_modelTemplateFilterText, StringComparison.OrdinalIgnoreCase))
                && (string.IsNullOrEmpty(_modelTemplateFilterAuthor) || _modelTemplateFilterAuthor.Equals(template.Author, StringComparison.OrdinalIgnoreCase))
                && (_modelTemplateFilterTemplateType is null || _modelTemplateFilterTemplateType == template.TemplateType);
        }


        private void ModelTemplateSort()
        {
            if (ModelTemplateCollectionView is null)
                return;

            ModelTemplateCollectionView.SortDescriptions.Clear();
            if (ModelTemplateSortProperty == "Status")
            {
                var inverseDirction = ModelTemplateSortDirection == ListSortDirection.Ascending ? ListSortDirection.Descending : ListSortDirection.Ascending;
                ModelTemplateCollectionView.SortDescriptions.Add(new SortDescription("IsInstalled", inverseDirction));
                ModelTemplateCollectionView.SortDescriptions.Add(new SortDescription("Name", ListSortDirection.Ascending));
                return;
            }

            ModelTemplateCollectionView.SortDescriptions.Add(new SortDescription(ModelTemplateSortProperty, ModelTemplateSortDirection));
        }


        private ModelTemplateViewModel _selectedModelTemplate;

        public ModelTemplateViewModel SelectedModelTemplate
        {
            get { return _selectedModelTemplate; }
            set { _selectedModelTemplate = value; NotifyPropertyChanged(); }
        }


        private ICollectionView _modelTemplateCollectionView;

        public ICollectionView ModelTemplateCollectionView
        {
            get { return _modelTemplateCollectionView; }
            set { _modelTemplateCollectionView = value; NotifyPropertyChanged(); }
        }






        private string _modelTemplateSortProperty;
        private ListSortDirection _modelTemplateSortDirection;
        private string _modelTemplateFilterText;
        private string _modelTemplateFilterAuthor;
        private List<string> _modelTemplateFilterAuthors;

        public string ModelTemplateSortProperty
        {
            get { return _modelTemplateSortProperty; }
            set { _modelTemplateSortProperty = value; NotifyPropertyChanged(); ModelTemplateSort(); }
        }

        public ListSortDirection ModelTemplateSortDirection
        {
            get { return _modelTemplateSortDirection; }
            set { _modelTemplateSortDirection = value; NotifyPropertyChanged(); ModelTemplateSort(); }
        }

        public string ModelTemplateFilterText
        {
            get { return _modelTemplateFilterText; }
            set { _modelTemplateFilterText = value; NotifyPropertyChanged(); ModelTemplateRefresh(); }
        }

        private ModelTemplateType? _modelTemplateFilterTemplateType;

        public ModelTemplateType? ModelTemplateFilterTemplateType
        {
            get { return _modelTemplateFilterTemplateType; }
            set { _modelTemplateFilterTemplateType = value; NotifyPropertyChanged(); ModelTemplateRefresh(); }
        }


        public string ModelTemplateFilterAuthor
        {
            get { return _modelTemplateFilterAuthor; }
            set { _modelTemplateFilterAuthor = value; NotifyPropertyChanged(); ModelTemplateRefresh(); }
        }

        public List<string> ModelTemplateFilterAuthors
        {
            get { return _modelTemplateFilterAuthors; }
            set { _modelTemplateFilterAuthors = value; NotifyPropertyChanged(); }
        }

        private LayoutViewType _modelTemplateLayoutView;

        public LayoutViewType ModelTemplateLayoutView
        {
            get { return _modelTemplateLayoutView; }
            set { _modelTemplateLayoutView = value; NotifyPropertyChanged(); }
        }







        private async Task InstallLocalStableDiffusionModel()
        {
            var modelTemplate = UISettings.Templates.FirstOrDefault(x => x.Name == _selectedModelTemplate.Name);
            if (modelTemplate == null)
                return; // TODO: Error

            var folderDialog = new System.Windows.Forms.FolderBrowserDialog();
            if (folderDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                if (_selectedModelTemplate.TemplateType == ModelTemplateType.Upscaler)
                {

                }
                else
                {
                    var stableDiffusionTemplate = _selectedModelTemplate.StableDiffusionTemplate;
                    if (stableDiffusionTemplate == null)
                        return; // TODO: Error

                    var modelSet = _modelFactory.CreateModelSet(modelTemplate.Name, folderDialog.SelectedPath, stableDiffusionTemplate.PipelineType, stableDiffusionTemplate.ModelType);
                    var isModelSetValid = modelSet.ModelConfigurations.All(x => File.Exists(x.OnnxModelPath));
                    if (isModelSetValid == false)
                        return; // TODO: Error

                    UISettings.StableDiffusionModelSets.Add(new StableDiffusionModelSetViewModel
                    {
                        Name = modelSet.Name,
                        ModelSet = modelSet
                    });
                }
            }

            // Update/Save
            UISettings.Templates.Remove(modelTemplate);
            modelTemplate.IsInstalled = true;
            UISettings.Templates.Add(modelTemplate);
            SelectedModelTemplate = modelTemplate;
            await SaveConfigurationFile();
        }


        private async Task UninstallStableDiffusionModel()
        {
            //TODO: Unload
            var modelSet = UISettings.StableDiffusionModelSets.FirstOrDefault(x => x.Name == SelectedModelTemplate.Name);
            UISettings.StableDiffusionModelSets.Remove(modelSet);

            var modelTemplate = UISettings.Templates.FirstOrDefault(x => x.Name == _selectedModelTemplate.Name);
            if (modelTemplate == null)
                return; // TODO: Error

            if (modelTemplate.IsUserTemplate)
                UISettings.Templates.Remove(modelTemplate);

            UISettings.Templates.Remove(modelTemplate);
            modelTemplate.IsInstalled = false;
            UISettings.Templates.Add(modelTemplate);

            SelectedModelTemplate = modelTemplate;
            await SaveConfigurationFile();
        }


        private async Task RenameStableDiffusionModel()
        {
            var invalidNames = UISettings.Templates.Select(x => x.Name).ToList();
            var addModelDialog = _dialogService.GetDialog<TextInputDialog>();
            if (addModelDialog.ShowDialog("Rename Model", "New Name", 1, 50, invalidNames))
            {
                var modelSet = UISettings.StableDiffusionModelSets.FirstOrDefault(x => x.Name == _selectedModelTemplate.Name);
                modelSet.Name = addModelDialog.TextResult;
                modelSet.ModelSet.Name = addModelDialog.TextResult;

                var modelTemplate = UISettings.Templates.FirstOrDefault(x => x.Name == _selectedModelTemplate.Name);
                if (modelTemplate == null)
                    return; // TODO: Error

                UISettings.Templates.Remove(modelTemplate);
                if (!modelTemplate.IsUserTemplate)
                {
                    modelTemplate.IsInstalled = false;
                    var userTemplate = modelTemplate with
                    {
                        IsInstalled = true,
                        IsUserTemplate = true,
                        Name = addModelDialog.TextResult
                    };
                    UISettings.Templates.Add(modelTemplate);
                    UISettings.Templates.Add(userTemplate);
                    SelectedModelTemplate = userTemplate;
                }
                else
                {
                    modelTemplate.Name = addModelDialog.TextResult;
                    UISettings.Templates.Add(modelTemplate);
                }
                await SaveConfigurationFile();
            }
        }
    }

    public enum LayoutViewType
    {
        Tile = 0,
        List = 1
    }

    public enum ModelTemplateType
    {
        StableDiffusion = 0,
        StableDiffusionXL = 1,
        LatentConsistency = 10,
        LatentConsistencyXL = 11,
        InstaFlow = 30,
        Upscaler = 100,
        Other = 255
    }


    public class UpscaleModelSetViewModel
    {
        public string Name { get; set; }
        public UpscaleModelSet ModelSet { get; set; }
    }


    public record ModelTemplateViewModel : INotifyPropertyChanged
    {
        private string _name;

        public string Name
        {
            get { return _name; }
            set { _name = value; NotifyPropertyChanged(); }
        }


        private string _imageIcon;

        public string ImageIcon
        {
            get { return _imageIcon; }
            set { _imageIcon = value; }
        }


        private string _author;

        public string Author
        {
            get { return _author; }
            set { _author = value; }
        }


        private ModelTemplateType _templateType;

        public ModelTemplateType TemplateType
        {
            get { return _templateType; }
            set { _templateType = value; }
        }



        private bool _isInstalled;

        [JsonIgnore]
        public bool IsInstalled
        {
            get { return _isInstalled; }
            set { _isInstalled = value; NotifyPropertyChanged(); }
        }


        private bool _isDownloading;

        [JsonIgnore]
        public bool IsDownloading
        {
            get { return _isDownloading; }
            set { _isDownloading = value; }
        }

        private int _progressValue;

        [JsonIgnore]
        public int ProgressValue
        {
            get { return _progressValue; }
            set { _progressValue = value; NotifyPropertyChanged(); }
        }

        private string _progressText;

        [JsonIgnore]
        public string ProgressText
        {
            get { return _progressText; }
            set { _progressText = value; NotifyPropertyChanged(); }
        }

        public bool IsUserTemplate { get; set; }


        public StableDiffusionModelTemplate StableDiffusionTemplate { get; set; }


        #region INotifyPropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;

        public void NotifyPropertyChanged([CallerMemberName] string property = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(property));
        }
        #endregion
    }

    public record StableDiffusionModelTemplate(DiffuserPipelineType PipelineType, ModelType ModelType);

}
