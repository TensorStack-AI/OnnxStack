using Microsoft.Extensions.Logging;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Dialogs;
using OnnxStack.UI.Models;
using OnnxStack.UI.Services;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.Json.Serialization;
using System.Threading;
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
        private readonly IModelDownloadService _modelDownloadService;

        private ModelTemplateViewModel _selectedModelTemplate;
        private ICollectionView _modelTemplateCollectionView;

        private string _modelTemplateFilterText;
        private string _modelTemplateFilterAuthor;
        private string _modelTemplateFilterTemplateType;
        private List<string> _modelTemplateFilterAuthors;
        private ModelTemplateStatusFilter _modelTemplateFilterStatus;

        private LayoutViewType _modelTemplateLayoutView = LayoutViewType.TileSmall;
        private string _modelTemplateSortProperty;
        private ListSortDirection _modelTemplateSortDirection;


        /// <summary>
        /// Initializes a new instance of the <see cref="ModelSettingsView"/> class.
        /// </summary>
        public ModelSettingsView()
        {
            if (!DesignerProperties.GetIsInDesignMode(this))
            {
                _logger = App.GetService<ILogger<ModelSettingsView>>();
                _modelFactory = App.GetService<IModelFactory>();
                _dialogService = App.GetService<IDialogService>();
                _modelDownloadService = App.GetService<IModelDownloadService>();
            }


            UpdateModelCommand = new AsyncRelayCommand(UpdateModel);
            UpdateModelAdvancedCommand = new AsyncRelayCommand(UpdateModelAdvanced);

            RemoveModelCommand = new AsyncRelayCommand(RemoveModel);
            InstallModelCommand = new AsyncRelayCommand(InstallModel);
            UninstallModelCommand = new AsyncRelayCommand(UninstallModel);
            DownloadModelCommand = new AsyncRelayCommand<bool>(DownloadModel);
            DownloadModelCancelCommand = new AsyncRelayCommand(DownloadModelCancel);
            ModelTemplateFilterResetCommand = new AsyncRelayCommand(ModelTemplateFilterReset);
            UpdateModelMetadataCommand = new AsyncRelayCommand(UpdateModelMetadata);
            ViewModelMetadataCommand = new AsyncRelayCommand(ViewModelMetadata);

            HyperLinkNavigateCommand = new AsyncRelayCommand<string>(HyperLinkNavigate);
            ModelTemplateLayoutCommand = new AsyncRelayCommand<LayoutViewType>(ModelTemplateLayout);

            AddUpscaleModelCommand = new AsyncRelayCommand(AddUpscaleModel);
            AddStableDiffusionModelCommand = new AsyncRelayCommand(AddStableDiffusionModel);
            InitializeComponent();
        }



        private Task HyperLinkNavigate(string link)
        {
            Process.Start(new ProcessStartInfo(link) { UseShellExecute = true });
            return Task.CompletedTask;
        }


        public AsyncRelayCommand UpdateModelCommand { get; }
        public AsyncRelayCommand UpdateModelAdvancedCommand { get; }
        public AsyncRelayCommand RemoveModelCommand { get; }
        public AsyncRelayCommand InstallModelCommand { get; }
        public AsyncRelayCommand UninstallModelCommand { get; }
        public AsyncRelayCommand<bool> DownloadModelCommand { get; }
        public AsyncRelayCommand DownloadModelCancelCommand { get; }
        public AsyncRelayCommand ModelTemplateFilterResetCommand { get; }
        public AsyncRelayCommand<string> HyperLinkNavigateCommand { get; }
        public AsyncRelayCommand AddUpscaleModelCommand { get; }
        public AsyncRelayCommand AddStableDiffusionModelCommand { get; }
        public AsyncRelayCommand<LayoutViewType> ModelTemplateLayoutCommand { get; }
        public AsyncRelayCommand UpdateModelMetadataCommand { get; }
        public AsyncRelayCommand ViewModelMetadataCommand { get; }

        public ModelTemplateViewModel SelectedModelTemplate
        {
            get { return _selectedModelTemplate; }
            set { _selectedModelTemplate = value; NotifyPropertyChanged(); }
        }

        public ICollectionView ModelTemplateCollectionView
        {
            get { return _modelTemplateCollectionView; }
            set { _modelTemplateCollectionView = value; NotifyPropertyChanged(); }
        }

        public string ModelTemplateFilterText
        {
            get { return _modelTemplateFilterText; }
            set { _modelTemplateFilterText = value; NotifyPropertyChanged(); ModelTemplateRefresh(); }
        }

        public string ModelTemplateFilterTemplateType
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



        public ModelTemplateStatusFilter ModelTemplateFilterStatus
        {
            get { return _modelTemplateFilterStatus; }
            set { _modelTemplateFilterStatus = value; NotifyPropertyChanged(); ModelTemplateRefresh(); }
        }




        public LayoutViewType ModelTemplateLayoutView
        {
            get { return _modelTemplateLayoutView; }
            set { _modelTemplateLayoutView = value; NotifyPropertyChanged(); }
        }

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

        public OnnxStackUIConfig UISettings
        {
            get { return (OnnxStackUIConfig)GetValue(UISettingsProperty); }
            set { SetValue(UISettingsProperty, value); }
        }
        public static readonly DependencyProperty UISettingsProperty =
            DependencyProperty.Register("UISettings", typeof(OnnxStackUIConfig), typeof(ModelSettingsView), new PropertyMetadata((d, e) =>
            {
                if (d is ModelSettingsView control && e.NewValue is OnnxStackUIConfig)
                    control.InitializeTemplates();
            }));


        public Task NavigateAsync(ImageResult imageResult)
        {
            throw new NotImplementedException();
        }


        private async Task RemoveModel()
        {
            var modelTemplate = UISettings.Templates.FirstOrDefault(x => x.Name == _selectedModelTemplate.Name);
            if (modelTemplate == null)
                return;

            if (modelTemplate.Category == ModelTemplateCategory.Upscaler)
                await RemoveUpscaleModel(modelTemplate);
            else if (modelTemplate.Category == ModelTemplateCategory.StableDiffusion)
                await RemoveStableDiffusionModel(modelTemplate);
        }


        private async Task InstallModel()
        {
            var modelTemplate = UISettings.Templates.FirstOrDefault(x => x.Name == _selectedModelTemplate.Name);
            if (modelTemplate == null)
                return;

            if (modelTemplate.Category == ModelTemplateCategory.Upscaler)
                await InstallUpscaleModel(modelTemplate);
            else if (modelTemplate.Category == ModelTemplateCategory.StableDiffusion)
                await InstallStableDiffusionModel(modelTemplate);
        }


        private async Task UninstallModel()
        {
            var modelTemplate = UISettings.Templates.FirstOrDefault(x => x.Name == _selectedModelTemplate.Name);
            if (modelTemplate == null)
                return;

            if (modelTemplate.Category == ModelTemplateCategory.Upscaler)
                await UninstallUpscaleModel(modelTemplate);
            else if (modelTemplate.Category == ModelTemplateCategory.StableDiffusion)
                await UninstallStableDiffusionModel(modelTemplate);
        }


        private async Task UpdateModel()
        {
            var modelTemplate = UISettings.Templates.FirstOrDefault(x => x.IsUserTemplate && x.Name == _selectedModelTemplate.Name);
            if (modelTemplate == null)
                return;

            if (modelTemplate.Category == ModelTemplateCategory.Upscaler)
                await UpdateUpscaleModel(modelTemplate);
            else if (modelTemplate.Category == ModelTemplateCategory.StableDiffusion)
                await UpdateStableDiffusionModel(modelTemplate);
        }

        private async Task UpdateModelAdvanced()
        {
            var modelTemplate = UISettings.Templates.FirstOrDefault(x => x.IsUserTemplate && x.Name == _selectedModelTemplate.Name);
            if (modelTemplate == null)
                return;

            if (modelTemplate.Category == ModelTemplateCategory.Upscaler)
                await UpdateUpscaleModelAdvanced(modelTemplate);
            else if (modelTemplate.Category == ModelTemplateCategory.StableDiffusion)
                await UpdateStableDiffusionModelAdvanced(modelTemplate);
        }



        private Task DownloadModel(bool isRepostitoryClone)
        {
            var modelTemplate = UISettings.Templates.FirstOrDefault(x => x.Name == _selectedModelTemplate.Name);
            if (modelTemplate == null)
                return Task.CompletedTask;

            var folderDialog = new System.Windows.Forms.FolderBrowserDialog();
            if (folderDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                var repositoryUrl = modelTemplate.Repository;
                var outputDirectory = Path.Combine(folderDialog.SelectedPath, repositoryUrl.Split('/').LastOrDefault());
                modelTemplate.IsDownloading = true;
                modelTemplate.CancellationTokenSource = new CancellationTokenSource();

                // Download File, do not await
                _ = Task.Factory.StartNew(async () =>
                {
                    modelTemplate.ErrorMessage = null;
                    modelTemplate.ProgressValue = 1;
                    modelTemplate.ProgressText = $"Starting Download...";
                    Action<string, double, double> progress = (f, fp, tp) =>
                    {
                        modelTemplate.ProgressText = $"{f}";
                        modelTemplate.ProgressValue = tp;
                    };
                    try
                    {
                        var isDownloadComplete = !isRepostitoryClone
                           ? await _modelDownloadService.DownloadHttpAsync(modelTemplate.RepositoryFiles, outputDirectory, progress, modelTemplate.CancellationTokenSource.Token)
                           : await _modelDownloadService.DownloadRepositoryAsync(modelTemplate.RepositoryClone, outputDirectory, progress, modelTemplate.CancellationTokenSource.Token);
                        App.UIInvoke(async () =>
                        {
                            if (isDownloadComplete)
                            {
                                if (modelTemplate.Category == ModelTemplateCategory.Upscaler)
                                    await DownloadUpscaleModelComplete(modelTemplate, outputDirectory);
                                if (modelTemplate.Category == ModelTemplateCategory.StableDiffusion)
                                    await DownloadStableDiffusionModelComplete(modelTemplate, outputDirectory);
                            }

                            modelTemplate.IsDownloading = false;
                        });
                    }
                    catch (Exception ex)
                    {
                        modelTemplate.IsDownloading = false;
                        modelTemplate.ProgressText = null;
                        modelTemplate.ProgressValue = 0;
                        modelTemplate.ErrorMessage = ex.Message;
                    }
                });
            }
            return Task.CompletedTask;
        }

        private Task DownloadModelCancel()
        {
            _selectedModelTemplate?.CancellationTokenSource?.Cancel();
            return Task.CompletedTask;
        }

        private Task ModelTemplateLayout(LayoutViewType layoutViewType)
        {
            ModelTemplateLayoutView = layoutViewType;
            return Task.CompletedTask;
        }

        private async Task UpdateModelMetadata()
        {
            var updateMetadataDialog = _dialogService.GetDialog<UpdateModelMetadataDialog>();
            if (updateMetadataDialog.ShowDialog(_selectedModelTemplate))
            {
                await SaveConfigurationFile();
            }
        }

        private Task ViewModelMetadata()
        {
            var viewMetadataDialog = _dialogService.GetDialog<ViewModelMetadataDialog>();
            viewMetadataDialog.ShowDialog(_selectedModelTemplate);
            return Task.CompletedTask;
        }

        private Task<bool> SaveConfigurationFile()
        {
            try
            {
                ConfigManager.SaveConfiguration(UISettings);
                ModelTemplateRefresh();
                return Task.FromResult(true);
            }
            catch (Exception ex)
            {
                _logger.LogError($"Error saving configuration file, {ex.Message}");
                return Task.FromResult(false);
            }
        }


        #region ModelTemplate

        private void InitializeTemplates()
        {
            if (ModelTemplateCollectionView != null)
                ModelTemplateCollectionView.CollectionChanged -= ModelTemplateCollectionView_CollectionChanged;

            foreach (var template in UISettings.Templates.Where(x => x.IsUserTemplate))
            {
                template.IsInstalled = UISettings.UpscaleModelSets.Any(x => x.Name == template.Name)
                                    || UISettings.StableDiffusionModelSets.Any(x => x.Name == template.Name);
            }

            ModelTemplateFilterAuthors = UISettings.Templates
               .Where(x => !string.IsNullOrEmpty(x.Author))
               .Select(x => x.Author)
               .Distinct()
               .OrderBy(x => x)
               .ToList();
            ModelTemplateFilterAuthors.Insert(0, "All");
            ModelTemplateFilterAuthor = "All";
            ModelTemplateCollectionView = new ListCollectionView(UISettings.Templates);
            ModelTemplateSort();
            ModelTemplateCollectionView.Filter = ModelTemplateFilter;
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
                && (_modelTemplateFilterAuthor == "All" || _modelTemplateFilterAuthor.Equals(template.Author, StringComparison.OrdinalIgnoreCase))
                && (_modelTemplateFilterTemplateType is null || _modelTemplateFilterTemplateType == template.Template)
                && (_modelTemplateFilterStatus == ModelTemplateStatusFilter.All
                                                       || (_modelTemplateFilterStatus == ModelTemplateStatusFilter.Installed && template.IsInstalled && template.IsUserTemplate)
                                                       || (_modelTemplateFilterStatus == ModelTemplateStatusFilter.Uninstalled && !template.IsInstalled && template.IsUserTemplate)
                                                       || (_modelTemplateFilterStatus == ModelTemplateStatusFilter.Template && !template.IsUserTemplate));
        }

        private Task ModelTemplateFilterReset()
        {
            ModelTemplateFilterAuthor = "All";
            ModelTemplateFilterStatus = default;
            ModelTemplateFilterTemplateType = null;
            ModelTemplateFilterText = null;
            return Task.CompletedTask;
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
                ModelTemplateCollectionView.SortDescriptions.Add(new SortDescription("IsUserTemplate", ListSortDirection.Ascending));
                ModelTemplateCollectionView.SortDescriptions.Add(new SortDescription("Rank", ListSortDirection.Descending));
                ModelTemplateCollectionView.SortDescriptions.Add(new SortDescription("Name", ListSortDirection.Ascending));
                return;
            }
            ModelTemplateCollectionView.SortDescriptions.Add(new SortDescription(ModelTemplateSortProperty, ModelTemplateSortDirection));
        }

        #endregion

        #region StableDiffusion Model

        private async Task AddStableDiffusionModel()
        {
            var addModelDialog = _dialogService.GetDialog<AddModelDialog>();
            if (addModelDialog.ShowDialog())
            {
                 await InstallStableDiffusionModel(addModelDialog.ModelTemplate, addModelDialog.ModelSetResult);
            }
        }

        private async Task RemoveStableDiffusionModel(ModelTemplateViewModel modelTemplate)
        {
            if (!modelTemplate.IsUserTemplate)
                return; // Cant remove Templates

            var modelSet = UISettings.StableDiffusionModelSets.FirstOrDefault(x => x.Name == modelTemplate.Name);
            if (modelSet.IsLoaded)
            {
                MessageBox.Show("Please unload model before uninstalling", "Model In Use", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            UISettings.StableDiffusionModelSets.Remove(modelSet);
            UISettings.Templates.Remove(modelTemplate);
            await SaveConfigurationFile();
        }


        private async Task InstallStableDiffusionModel(ModelTemplateViewModel modelTemplate)
        {
            var addModelDialog = _dialogService.GetDialog<AddModelDialog>();
            if (!addModelDialog.ShowDialog(modelTemplate))
                return; // User Canceled

            await InstallStableDiffusionModel(modelTemplate, addModelDialog.ModelSetResult);
        }


        private async Task InstallStableDiffusionModel(ModelTemplateViewModel modelTemplate, StableDiffusionModelSet modelSetResult)
        {
            if (modelTemplate.IsUserTemplate)
            {
                modelTemplate.IsInstalled = true;
            }
            else
            {
                var newModelTemplate = new ModelTemplateViewModel
                {
                    Name = modelSetResult.Name,
                    Category = ModelTemplateCategory.StableDiffusion,
                    IsInstalled = true,
                    IsUserTemplate = true,
                    ImageIcon = string.Empty,// modelTemplate.ImageIcon,
                    Author = "Unknown", //modelTemplate.Author,
                    Template = modelTemplate.Template,
                    StableDiffusionTemplate = modelTemplate.StableDiffusionTemplate with { },
                };
                UISettings.Templates.Add(newModelTemplate);
                SelectedModelTemplate = newModelTemplate;
            }

            UISettings.StableDiffusionModelSets.Add(new StableDiffusionModelSetViewModel
            {
                Name = modelSetResult.Name,
                ModelSet = modelSetResult
            });
            await SaveConfigurationFile();
        }

        private async Task UninstallStableDiffusionModel(ModelTemplateViewModel modelTemplate)
        {
            var modelSet = UISettings.StableDiffusionModelSets.FirstOrDefault(x => x.Name == modelTemplate.Name);
            if (modelSet.IsLoaded)
            {
                MessageBox.Show("Please unload model before uninstalling", "Model In Use", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            UISettings.StableDiffusionModelSets.Remove(modelSet);
            modelTemplate.IsInstalled = false;
            await SaveConfigurationFile();
        }

        private async Task UpdateStableDiffusionModel(ModelTemplateViewModel modelTemplate)
        {
            var stableDiffusionModel = UISettings.StableDiffusionModelSets.FirstOrDefault(x => x.Name == modelTemplate.Name);
            if (stableDiffusionModel.IsLoaded)
            {
                MessageBox.Show("Please unload model before updating", "Model In Use", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }


            var updateModelDialog = _dialogService.GetDialog<UpdateModelSettingsDialog>();
            if (updateModelDialog.ShowDialog(stableDiffusionModel.ModelSet))
            {
                var modelSet = updateModelDialog.ModelSetResult;
                stableDiffusionModel.ModelSet = modelSet;
                stableDiffusionModel.Name = modelSet.Name;
                SelectedModelTemplate.Name = modelSet.Name;
                UISettings.StableDiffusionModelSets.ForceNotifyCollectionChanged();
                await SaveConfigurationFile();
            }
        }


        private async Task UpdateStableDiffusionModelAdvanced(ModelTemplateViewModel modelTemplate)
        {
            var stableDiffusionModel = UISettings.StableDiffusionModelSets.FirstOrDefault(x => x.Name == modelTemplate.Name);
            if (stableDiffusionModel.IsLoaded)
            {
                MessageBox.Show("Please unload model before updating", "Model In Use", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            var updateModelDialog = _dialogService.GetDialog<UpdateModelDialog>();
            if (updateModelDialog.ShowDialog(stableDiffusionModel.ModelSet))
            {
                var modelSet = updateModelDialog.ModelSetResult;
                stableDiffusionModel.ModelSet = modelSet;
                stableDiffusionModel.Name = modelSet.Name;
                SelectedModelTemplate.Name = modelSet.Name;
                UISettings.StableDiffusionModelSets.ForceNotifyCollectionChanged();
                await SaveConfigurationFile();
            }
        }

        private async Task DownloadStableDiffusionModelComplete(ModelTemplateViewModel modelTemplate, string outputDirectory)
        {
            var modelSet = _modelFactory.CreateStableDiffusionModelSet(modelTemplate.Name, outputDirectory, modelTemplate.StableDiffusionTemplate);
            var isModelSetValid = modelSet.ModelConfigurations.All(x => File.Exists(x.OnnxModelPath));
            if (!isModelSetValid)
            {
                // Error, Invalid modelset after download
                modelTemplate.IsDownloading = false;
                modelTemplate.ErrorMessage = "Error: Download completed but ModelSet is invalid";
                return;
            }

            UISettings.StableDiffusionModelSets.Add(new StableDiffusionModelSetViewModel
            {
                Name = modelSet.Name,
                ModelSet = modelSet
            });

            // Update/Save
            modelTemplate.IsInstalled = true;
            await SaveConfigurationFile();
        }

        #endregion

        #region Upscale Model

        private async Task AddUpscaleModel()
        {
            var addModelDialog = _dialogService.GetDialog<AddUpscaleModelDialog>();
            if (addModelDialog.ShowDialog())
            {
                await InstallUpscaleModel(addModelDialog.ModelTemplate, addModelDialog.ModelSetResult);
            }
        }

        private async Task UpdateUpscaleModel(ModelTemplateViewModel modelTemplate)
        {
            var upscaleModel = UISettings.UpscaleModelSets.FirstOrDefault(x => x.Name == modelTemplate.Name);
            if (upscaleModel.IsLoaded)
            {
                MessageBox.Show("Please unload model before updating", "Model In Use", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }


            var updateModelDialog = _dialogService.GetDialog<UpdateUpscaleModelSettingsDialog>();
            if (updateModelDialog.ShowDialog(upscaleModel.ModelSet))
            {
                var modelSet = updateModelDialog.ModelSetResult;
                upscaleModel.ModelSet = modelSet;
                upscaleModel.Name = modelSet.Name;
                SelectedModelTemplate.Name = modelSet.Name;
                UISettings.UpscaleModelSets.ForceNotifyCollectionChanged();
                await SaveConfigurationFile();
            }
        }

        private async Task UpdateUpscaleModelAdvanced(ModelTemplateViewModel modelTemplate)
        {
            var upscaleModel = UISettings.UpscaleModelSets.FirstOrDefault(x => x.Name == modelTemplate.Name);
            if (upscaleModel.IsLoaded)
            {
                MessageBox.Show("Please unload model before updating", "Model In Use", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            var updateModelDialog = _dialogService.GetDialog<UpdateUpscaleModelDialog>();
            if (updateModelDialog.ShowDialog(upscaleModel.ModelSet))
            {
                var modelSet = updateModelDialog.ModelSetResult;
                upscaleModel.ModelSet = modelSet;
                upscaleModel.Name = modelSet.Name;
                SelectedModelTemplate.Name = modelSet.Name;
                UISettings.UpscaleModelSets.ForceNotifyCollectionChanged();
                await SaveConfigurationFile();
            }
        }

        private async Task RemoveUpscaleModel(ModelTemplateViewModel modelTemplate)
        {
            if (!modelTemplate.IsUserTemplate)
                return; // Cant remove Templates
        
            var modelSet = UISettings.UpscaleModelSets.FirstOrDefault(x => x.Name == modelTemplate.Name);
            if (modelSet.IsLoaded)
            {
                MessageBox.Show("Please unload model before uninstalling", "Model In Use", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            UISettings.UpscaleModelSets.Remove(modelSet);
            UISettings.Templates.Remove(modelTemplate);
            await SaveConfigurationFile();
        }


        private async Task InstallUpscaleModel(ModelTemplateViewModel modelTemplate)
        {
            var addModelDialog = _dialogService.GetDialog<AddUpscaleModelDialog>();
            if (!addModelDialog.ShowDialog(modelTemplate))
                return; // User Canceled

            await InstallUpscaleModel(modelTemplate, addModelDialog.ModelSetResult);
        }

        private async Task InstallUpscaleModel(ModelTemplateViewModel modelTemplate, UpscaleModelSet modelSetResult)
        {
            if (modelTemplate.IsUserTemplate)
            {
                modelTemplate.IsInstalled = true;
            }
            else
            {
                var newModelTemplate = new ModelTemplateViewModel
                {
                    Name = modelSetResult.Name,
                    Category = ModelTemplateCategory.Upscaler,
                    IsInstalled = true,
                    IsUserTemplate = true,
                    ImageIcon = string.Empty,// modelTemplate.ImageIcon,
                    Author = "Unknown", //modelTemplate.Author,
                    Template = modelTemplate.Template,
                    UpscaleTemplate = modelTemplate.UpscaleTemplate with { },
                };
                UISettings.Templates.Add(newModelTemplate);
                SelectedModelTemplate = newModelTemplate;
            }

            UISettings.UpscaleModelSets.Add(new UpscaleModelSetViewModel
            {
                Name = modelSetResult.Name,
                ModelSet = modelSetResult
            });
            await SaveConfigurationFile();
        }


        private async Task UninstallUpscaleModel(ModelTemplateViewModel modelTemplate)
        {
            var modelSet = UISettings.UpscaleModelSets.FirstOrDefault(x => x.Name == modelTemplate.Name);
            if (modelSet.IsLoaded)
            {
                MessageBox.Show("Please unload model before uninstalling", "Model In Use", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            UISettings.UpscaleModelSets.Remove(modelSet);
            modelTemplate.IsInstalled = false;
            await SaveConfigurationFile();
        }

        private async Task DownloadUpscaleModelComplete(ModelTemplateViewModel modelTemplate, string outputDirectory)
        {
            var modelSet = _modelFactory.CreateUpscaleModelSet(modelTemplate.Name, outputDirectory, modelTemplate.UpscaleTemplate);
            var isModelSetValid = modelSet.ModelConfigurations.All(x => File.Exists(x.OnnxModelPath));
            if (!isModelSetValid)
            {
                // Error, Invalid modelset after download
                modelTemplate.IsDownloading = false;
                modelTemplate.ErrorMessage = "Error: Download completed but ModelSet is invalid";
                return;
            }

            UISettings.UpscaleModelSets.Add(new UpscaleModelSetViewModel
            {
                Name = modelSet.Name,
                ModelSet = modelSet
            });

            // Update/Save
            modelTemplate.IsInstalled = true;
            await SaveConfigurationFile();
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

    public enum LayoutViewType
    {
        TileLarge = 0,
        TileSmall = 1,
        List = 2
    }


    public enum ModelTemplateCategory
    {
        StableDiffusion = 0,
        Upscaler = 1
    }

    public enum ModelTemplateStatusFilter
    {
        All = 0,
        Installed = 1,
        Template = 2,
        Uninstalled = 3
    }


    public class ModelTemplateViewModel : INotifyPropertyChanged
    {
        private string _name;
        private string _imageIcon;
        private string _author;

        private bool _isInstalled;
        private bool _isDownloading;
        private double _progressValue;
        private string _progressText;
        private string _errorMessage;

        public string Name
        {
            get { return _name; }
            set { _name = value; NotifyPropertyChanged(); }
        }

        public string ImageIcon
        {
            get { return _imageIcon; }
            set { _imageIcon = value; NotifyPropertyChanged(); }
        }

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

        public int Rank { get; set; }
        public bool IsUserTemplate { get; set; }

        public string Template { get; set; }

        public ModelTemplateCategory Category { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public UpscaleModelTemplate UpscaleTemplate { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public StableDiffusionModelTemplate StableDiffusionTemplate { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public string Website { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public string Repository { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public string RepositoryBranch { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public List<string> RepositoryFiles { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public List<string> PreviewImages { get; set; }

        [JsonIgnore]
        public string RepositoryClone
        {
            get
            {
                return string.IsNullOrEmpty(RepositoryBranch)
                ? Repository
                : $"{Repository} -b {RepositoryBranch}";
            }
        }

        [JsonIgnore]
        public bool IsInstalled
        {
            get { return _isInstalled; }
            set { _isInstalled = value; NotifyPropertyChanged(); }
        }

        [JsonIgnore]
        public bool IsDownloading
        {
            get { return _isDownloading; }
            set { _isDownloading = value; NotifyPropertyChanged(); }
        }

        [JsonIgnore]
        public double ProgressValue
        {
            get { return _progressValue; }
            set { _progressValue = value; NotifyPropertyChanged(); }
        }

        [JsonIgnore]
        public string ProgressText
        {
            get { return _progressText; }
            set { _progressText = value; NotifyPropertyChanged(); }
        }

        [JsonIgnore]
        public string ErrorMessage
        {
            get { return _errorMessage; }
            set { _errorMessage = value; NotifyPropertyChanged(); }
        }

        [JsonIgnore]
        public bool IsRepositoryCloneEnabled => !string.IsNullOrEmpty(Repository);

        [JsonIgnore]
        public bool IsRepositoryDownloadEnabled => !RepositoryFiles.IsNullOrEmpty();

        [JsonIgnore]
        public CancellationTokenSource CancellationTokenSource { get; set; }



        #region INotifyPropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;

        public void NotifyPropertyChanged([CallerMemberName] string property = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(property));
        }
        #endregion
    }



}
