using Microsoft.Extensions.Logging;
using Microsoft.Win32;
using Models;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Dialogs;
using OnnxStack.UI.Models;
using OnnxStack.UI.Services;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace OnnxStack.UI.Views
{
    /// <summary>
    /// Interaction logic for ModelView.xaml
    /// </summary>
    public partial class ModelView : UserControl, INavigatable, INotifyPropertyChanged
    {
        private readonly ILogger<ModelView> _logger;
        private readonly string _defaultTokenizerPath;
        private readonly IDialogService _dialogService;
        private readonly IOnnxModelService _onnxModelService;
        private readonly IModelDownloadService _modelDownloadService;
        private readonly StableDiffusionConfig _stableDiffusionConfig;
        private readonly IStableDiffusionService _stableDiffusionService;

        private bool _isDownloading;
        private ModelSetViewModel _selectedModelSet;
        private ObservableCollection<ModelSetViewModel> _modelSets;
        private CancellationTokenSource _downloadCancellationTokenSource;


        /// <summary>
        /// Initializes a new instance of the <see cref="ModelView"/> class.
        /// </summary>
        public ModelView()
        {
            if (!DesignerProperties.GetIsInDesignMode(this))
            {
                _logger = App.GetService<ILogger<ModelView>>();
                _dialogService = App.GetService<IDialogService>();
                _onnxModelService = App.GetService<IOnnxModelService>();
                _stableDiffusionConfig = App.GetService<StableDiffusionConfig>();
                _stableDiffusionService = App.GetService<IStableDiffusionService>();
                _modelDownloadService = App.GetService<IModelDownloadService>();

                // Set Default tokenizer path
                var defaultTokenizerPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "cliptokenizer.onnx");
                if (File.Exists(defaultTokenizerPath))
                    _defaultTokenizerPath = defaultTokenizerPath;
            }

            SaveCommand = new AsyncRelayCommand(Save, CanExecuteSave);
            ResetCommand = new AsyncRelayCommand(Reset, CanExecuteReset);
            AddCommand = new AsyncRelayCommand(Add, CanExecuteAdd);
            CopyCommand = new AsyncRelayCommand(Copy, CanExecuteCopy);
            RemoveCommand = new AsyncRelayCommand(Remove, CanExecuteRemove);
            RenameCommand = new AsyncRelayCommand(Rename);
            InstallLocalCommand = new AsyncRelayCommand(InstallLocal, CanExecuteInstallLocal);
            InstallRemoteCommand = new AsyncRelayCommand(InstallRemote, CanExecuteInstallRemote);
            InstallRepositoryCommand = new AsyncRelayCommand(InstallRepository, CanExecuteInstallRepository);
            InstallCancelCommand = new AsyncRelayCommand(InstallCancel);
            ImportCommand = new AsyncRelayCommand(Import);
            ExportCommand = new AsyncRelayCommand(Export, CanExecuteExport);
            InitializeComponent();
        }

        public AsyncRelayCommand SaveCommand { get; }
        public AsyncRelayCommand ResetCommand { get; }
        public AsyncRelayCommand AddCommand { get; }
        public AsyncRelayCommand CopyCommand { get; }
        public AsyncRelayCommand RemoveCommand { get; }
        public AsyncRelayCommand RenameCommand { get; }
        public AsyncRelayCommand ImportCommand { get; }
        public AsyncRelayCommand ExportCommand { get; }
        public AsyncRelayCommand InstallLocalCommand { get; }
        public AsyncRelayCommand InstallRemoteCommand { get; }
        public AsyncRelayCommand InstallRepositoryCommand { get; }
        public AsyncRelayCommand InstallCancelCommand { get; }

        public OnnxStackUIConfig UISettings
        {
            get { return (OnnxStackUIConfig)GetValue(UISettingsProperty); }
            set { SetValue(UISettingsProperty, value); }
        }
        public static readonly DependencyProperty UISettingsProperty =
            DependencyProperty.Register("UISettings", typeof(OnnxStackUIConfig), typeof(ModelView), new PropertyMetadata(propertyChangedCallback: (s, e) =>
            {
                if (s is ModelView modelView && e.NewValue is OnnxStackUIConfig config)
                    modelView.Initialize();
            }));

        public ObservableCollection<ModelOptionsModel> ModelOptions
        {
            get { return (ObservableCollection<ModelOptionsModel>)GetValue(ModelOptionsProperty); }
            set { SetValue(ModelOptionsProperty, value); }
        }

        public static readonly DependencyProperty ModelOptionsProperty =
            DependencyProperty.Register("ModelOptions", typeof(ObservableCollection<ModelOptionsModel>), typeof(ModelView));

        public ObservableCollection<ModelSetViewModel> ModelSets
        {
            get { return _modelSets; }
            set { _modelSets = value; NotifyPropertyChanged(); }
        }

        public ModelSetViewModel SelectedModelSet
        {
            get { return _selectedModelSet; }
            set { _selectedModelSet = value; NotifyPropertyChanged(); }
        }

        public Task NavigateAsync(ImageResult imageResult)
        {
            return Task.CompletedTask;
        }


        #region Initialize/Reset

        /// <summary>
        /// Initializes the settings instance.
        /// </summary>
        private void Initialize()
        {
            ModelSets = new ObservableCollection<ModelSetViewModel>();
            foreach (var installedModel in _stableDiffusionConfig.OnnxModelSets.Select(CreateViewModel))
            {
                _logger.LogDebug($"Initialize ModelSet: {installedModel.Name}");

                // Find matching installed template
                var template = UISettings.ModelTemplates
                    .Where(x => x.Status == ModelTemplateStatus.Installed)
                    .FirstOrDefault(x => x.Name == installedModel.Name);

                // TODO: add extra template properties, images etc
                installedModel.ModelTemplate = template;
                ModelSets.Add(installedModel);

                installedModel.ResetChanges();
            }

            // Add any Active templates
            foreach (var templateModel in UISettings.ModelTemplates.Where(x => x.Status == ModelTemplateStatus.Active).Select(CreateViewModel))
            {
                _logger.LogDebug($"Initialize ModelTemplate: {templateModel.Name}");

                ModelSets.Add(templateModel);
            }
        }


        /// <summary>
        /// Resets the settings instance.
        /// </summary>
        /// <returns></returns>
        private Task Reset()
        {
            Initialize();
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can execute reset.
        /// </summary>
        /// <returns>
        ///   <c>true</c> if this instance can execute reset; otherwise, <c>false</c>.
        /// </returns>
        private bool CanExecuteReset()
        {
            return true;
        }

        #endregion

        #region Install


        /// <summary>
        /// Installs a local ModelSet.
        /// </summary>
        private async Task InstallLocal()
        {
            var folderDialog = new System.Windows.Forms.FolderBrowserDialog();
            if (folderDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                var template = UISettings.ModelTemplates.FirstOrDefault(x => x.Name == SelectedModelSet.Name);
                _logger.LogDebug($"Installing local ModelSet, ModelSet: {template.Name}, Directory: {folderDialog.SelectedPath}");
                if (SetModelPaths(SelectedModelSet, folderDialog.SelectedPath))
                {
                    SelectedModelSet.IsEnabled = true;
                    SelectedModelSet.IsInstalled = true;
                    SelectedModelSet.IsTemplate = false;
                    SelectedModelSet.DeviceId = UISettings.DefaultDeviceId;
                    SelectedModelSet.ExecutionMode = UISettings.DefaultExecutionMode;
                    SelectedModelSet.ExecutionProvider = UISettings.DefaultExecutionProvider;
                    SelectedModelSet.InterOpNumThreads = UISettings.DefaultInterOpNumThreads;
                    SelectedModelSet.IntraOpNumThreads = UISettings.DefaultIntraOpNumThreads;
                    SelectedModelSet.ModelTemplate = template;
                    await SaveModelAsync(SelectedModelSet);
                }
            }
        }


        /// <summary>
        /// Determines whether this instance can execute InstallLocal.
        /// </summary>
        /// <returns>
        ///   <c>true</c> if this instance can execute InstallLocal; otherwise, <c>false</c>.
        /// </returns>
        private bool CanExecuteInstallLocal()
        {
            return SelectedModelSet?.IsTemplate ?? false;
        }


        /// <summary>
        /// Installs a model from a remote location.
        /// </summary>
        private Task InstallRemote()
        {
            if (_isDownloading)
            {
                MessageBox.Show("There is already a model download in progress");
                return Task.CompletedTask;
            }

            var folderDialog = new System.Windows.Forms.FolderBrowserDialog();
            if (folderDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                var modelSet = ModelSets.FirstOrDefault(x => x.Name == SelectedModelSet.Name);
                var template = UISettings.ModelTemplates.FirstOrDefault(x => x.Name == modelSet.Name);
                var repositoryUrl = template.Repository;
                var modelDirectory = Path.Combine(folderDialog.SelectedPath, template.Repository.Split('/').LastOrDefault());

                _logger.LogDebug($"Download remote ModelSet, ModelSet: {template.Name}, Directory: {folderDialog.SelectedPath}");

                // Download File, do not await
                _ = DownloadRemote(modelSet, template, modelDirectory);
            }
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can execute InstallRemote.
        /// </summary>
        /// <returns>
        ///   <c>true</c> if this instance can execute InstallRemote; otherwise, <c>false</c>.
        /// </returns>
        private bool CanExecuteInstallRemote()
        {
            return !_isDownloading && (SelectedModelSet?.ModelTemplate?.ModelFiles?.Any() ?? false);
        }


        /// <summary>
        /// Installs a model from a git repository (GIT-LFS must be installed).
        /// </summary>
        private Task InstallRepository()
        {
            if (_isDownloading)
            {
                MessageBox.Show("There is already a model download in progress");
                return Task.CompletedTask;
            }

            var folderDialog = new System.Windows.Forms.FolderBrowserDialog();
            if (folderDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                var modelSet = ModelSets.FirstOrDefault(x => x.Name == SelectedModelSet.Name);
                var template = UISettings.ModelTemplates.FirstOrDefault(x => x.Name == modelSet.Name);
                var repositoryUrl = template.Repository;
                var modelDirectory = Path.Combine(folderDialog.SelectedPath, template.Repository.Split('/').LastOrDefault());

                _logger.LogDebug($"Download repository ModelSet, ModelSet: {template.Name}, Directory: {folderDialog.SelectedPath}");

                // Download File, do not await
                _ = DownloadRepository(modelSet, template, modelDirectory);
            }
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can execute InstallRepository.
        /// </summary>
        /// <returns>
        ///   <c>true</c> if this instance can execute InstallRepository; otherwise, <c>false</c>.
        /// </returns>
        private bool CanExecuteInstallRepository()
        {
            return !_isDownloading && !string.IsNullOrEmpty(SelectedModelSet?.ModelTemplate?.Repository); // and has git-lfs installed
        }



        /// <summary>
        /// Called when a model download is complete.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="modelDirectory">The model directory.</param>
        private async Task OnDownloadComplete(ModelSetViewModel modelSet, string modelDirectory)
        {
            _logger.LogDebug($"Download complete, ModelSet: {modelSet.Name}, Directory: {modelDirectory}");

            _isDownloading = false;
            modelSet.IsDownloading = false;
            if (!SetModelPaths(modelSet, modelDirectory))
            {
                _logger.LogError($"Failed to set model paths, ModelSet: {modelSet.Name}, Directory: {modelDirectory}");
                return;
            }

            modelSet.IsEnabled = true;
            modelSet.IsInstalled = true;
            modelSet.IsTemplate = false;
            modelSet.DeviceId = UISettings.DefaultDeviceId;
            modelSet.ExecutionMode = UISettings.DefaultExecutionMode;
            modelSet.ExecutionProvider = UISettings.DefaultExecutionProvider;
            modelSet.InterOpNumThreads = UISettings.DefaultInterOpNumThreads;
            modelSet.IntraOpNumThreads = UISettings.DefaultIntraOpNumThreads;
            await SaveModelAsync(modelSet);
            return;
        }



        /// <summary>
        /// Cancels the current install download.
        /// </summary>
        /// <returns></returns>
        private Task InstallCancel()
        {
            _logger.LogInformation($"Download canceled, ModelSet: {SelectedModelSet.Name}");
            if (_isDownloading)
                _downloadCancellationTokenSource?.Cancel();

            return Task.CompletedTask;
        }


        /// <summary>
        /// Downloads the remote model.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="template">The template.</param>
        /// <param name="outputDirectory">The output directory.</param>
        private async Task DownloadRemote(ModelSetViewModel modelSet, ModelConfigTemplate template, string outputDirectory)
        {
            try
            {
                _isDownloading = true;
                modelSet.IsDownloading = true;
                Action<string, double, double> progress = (f, fp, tp) =>
                {
                    modelSet.ProgessText = $"{f}";
                    modelSet.ProgressValue = tp;
                };


                _downloadCancellationTokenSource = new CancellationTokenSource();
                if (await _modelDownloadService.DownloadHttpAsync(template, outputDirectory, progress, _downloadCancellationTokenSource.Token))
                {
                    await OnDownloadComplete(modelSet, outputDirectory);
                }

            }
            catch (Exception ex)
            {
                _isDownloading = false;
                modelSet.IsDownloading = false;
                modelSet.ProgressValue = 0;
                modelSet.ProgessText = $"Error: {ex.Message}";
                _logger.LogError($"Error downloading remote ModelSet, ModelSet: {modelSet.Name}");
            }
        }


        /// <summary>
        /// Downloads the repository.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="template">The template.</param>
        /// <param name="outputDirectory">The output directory.</param>
        private async Task DownloadRepository(ModelSetViewModel modelSet, ModelConfigTemplate template, string outputDirectory)
        {
            try
            {
                _isDownloading = true;
                modelSet.IsDownloading = true;
                Action<string, double, double> progress = (f, fp, tp) =>
                {
                    modelSet.ProgessText = $"{f}";
                    modelSet.ProgressValue = tp;
                };

                _downloadCancellationTokenSource = new CancellationTokenSource();
                if (await _modelDownloadService.DownloadRepositoryAsync(template, outputDirectory, progress, _downloadCancellationTokenSource.Token))
                {
                    await OnDownloadComplete(modelSet, outputDirectory);
                }
            }
            catch (Exception ex)
            {
                _isDownloading = false;
                modelSet.IsDownloading = false;
                modelSet.ProgressValue = 0;
                modelSet.ProgessText = $"Error: {ex.Message}";
                _logger.LogError($"Error downloading repository ModelSet, ModelSet: {modelSet.Name}");
            }
        }


        /// <summary>
        /// Sets the model paths.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="modelDirectory">The model directory.</param>
        /// <returns></returns>
        private bool SetModelPaths(ModelSetViewModel modelSet, string modelDirectory)
        {
            var unetPath = Path.Combine(modelDirectory, "unet", "model.onnx");
            var tokenizerPath = Path.Combine(modelDirectory, "tokenizer", "model.onnx");
            var textEncoderPath = Path.Combine(modelDirectory, "text_encoder", "model.onnx");
            var vaeDecoder = Path.Combine(modelDirectory, "vae_decoder", "model.onnx");
            var vaeEncoder = Path.Combine(modelDirectory, "vae_encoder", "model.onnx");
            var tokenizer2Path = Path.Combine(modelDirectory, "tokenizer_2", "model.onnx");
            var textEncoder2Path = Path.Combine(modelDirectory, "text_encoder_2", "model.onnx");

            if (!File.Exists(tokenizerPath))
                tokenizerPath = _defaultTokenizerPath;
            if (!File.Exists(tokenizer2Path))
                tokenizer2Path = _defaultTokenizerPath;

            // Set Model Paths
            foreach (var modelConfig in modelSet.ModelFiles)
            {
                modelConfig.OnnxModelPath = modelConfig.Type switch
                {
                    OnnxModelType.Unet => unetPath,
                    OnnxModelType.Tokenizer => tokenizerPath,
                    OnnxModelType.TextEncoder => textEncoderPath,
                    OnnxModelType.VaeDecoder => vaeDecoder,
                    OnnxModelType.VaeEncoder => vaeEncoder,
                    OnnxModelType.Tokenizer2 => tokenizer2Path,
                    OnnxModelType.TextEncoder2 => textEncoder2Path,
                    _ => default
                };
            }
            return true;
        }


        #endregion

        #region Save


        /// <summary>
        /// Saves the settings.
        /// </summary>
        private async Task Save()
        {
            // Unload and Remove ModelSet
            await UnloadAndRemoveModelSetAsync(SelectedModelSet.Name);
            await SaveModelAsync(SelectedModelSet);
        }


        /// <summary>
        /// Determines whether this instance can execute save.
        /// </summary>
        /// <returns>
        ///   <c>true</c> if this instance can execute save; otherwise, <c>false</c>.
        /// </returns>
        private bool CanExecuteSave()
        {
            return SelectedModelSet?.HasChanges() ?? false;
        }


        /// <summary>
        /// Saves the model.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <returns></returns>
        private async Task<bool> SaveModelAsync(ModelSetViewModel modelSet)
        {
            _logger.LogInformation($"Saving configuration file...");

            // Create New ModelSet
            var newModelOption = CreateModelOptions(modelSet);
            newModelOption.InitBlankTokenArray();

            //Validate
            if (!ValidateModelSet(newModelOption))
            {
                // Retuen error
                _logger.LogError($"Failed to validate ModelSet, ModelSet: {modelSet.Name}");
                return false;
            }

            // Add to Config file
            _stableDiffusionConfig.OnnxModelSets.Add(newModelOption);

            // Update Templater if one was used
            UpdateTemplateStatus(newModelOption.Name, ModelTemplateStatus.Installed);

            // Save Config File
            if (!await SaveConfigurationFile())
                return false;


            // Update OnnxStack Service
            newModelOption.ApplyConfigurationOverrides();
            _onnxModelService.UpdateModelSet(newModelOption);

            // Add new ViewModel
            ModelOptions.Add(new ModelOptionsModel
            {
                Name = modelSet.Name,
                ModelOptions = newModelOption,
                IsEnabled = modelSet.IsEnabled,
            });

            // Force Rebind
            ModelOptions = new ObservableCollection<ModelOptionsModel>(ModelOptions);

            _logger.LogInformation($"Saving configuration complete.");
            return true;
        }

        #endregion

        #region Add/Copy/Remove/Rename


        /// <summary>
        /// Adds a custom model
        /// </summary>
        /// <returns></returns>
        private Task Add()
        {
            var invalidNames = ModelSets.Select(x => x.Name).ToList();
            var textInputDialog = _dialogService.GetDialog<AddModelDialog>();
            if (textInputDialog.ShowDialog("Add Model Set", invalidNames))
            {
                var pipeline = textInputDialog.PipelineType;
                var newModelTemplate = new ModelConfigTemplate
                {
                    Name = textInputDialog.TextResult,
                    Author = string.Empty,
                    Repository = string.Empty,
                    Description = string.Empty,
                    Status = ModelTemplateStatus.Active,
                    ImageIcon = string.Empty,
                    Images = Enumerable.Range(0, 6).Select(x => string.Empty).ToList(),

                    // TODO: Select pipleine in dialog, then setting any required bits
                    PipelineType = pipeline,
                    ScaleFactor = pipeline == DiffuserPipelineType.StableDiffusionXL ? 0.13025f : 0.18215f,
                    TokenizerLimit = 77,
                    PadTokenId = pipeline == DiffuserPipelineType.StableDiffusionXL ? 1 : 49407,
                    EmbeddingsLength = 768,
                    DualEmbeddingsLength = 1280,
                    IsDualTokenizer = pipeline == DiffuserPipelineType.StableDiffusionXL,
                    BlankTokenId = 49407,
                    Diffusers = Enum.GetValues<DiffuserType>().ToList(),
                };

                var modelSet = CreateViewModel(newModelTemplate);
                ModelSets.Add(modelSet);
                SelectedModelSet = modelSet;
                UISettings.ModelTemplates.Add(newModelTemplate);
            }
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can execute add.
        /// </summary>
        /// <returns>
        ///   <c>true</c> if this instance can execute add; otherwise, <c>false</c>.
        /// </returns>
        private bool CanExecuteAdd()
        {
            return true;
        }



        /// <summary>
        /// Copies a modelSet.
        /// </summary>
        /// <returns></returns>
        private Task Copy()
        {
            var invalidNames = ModelSets.Select(x => x.Name).ToList();
            var textInputDialog = _dialogService.GetDialog<TextInputDialog>();
            if (textInputDialog.ShowDialog("Copy Model Set", "New Name", 1, 30, invalidNames))
            {
                var newModelSet = SelectedModelSet.IsTemplate
                     ? CreateViewModel(UISettings.ModelTemplates.FirstOrDefault(x => x.Name == SelectedModelSet.Name))
                     : CreateViewModel(_stableDiffusionConfig.OnnxModelSets.FirstOrDefault(x => x.Name == SelectedModelSet.Name));

                newModelSet.IsEnabled = false;
                newModelSet.IsTemplate = false;
                newModelSet.IsInstalled = false;
                newModelSet.Name = textInputDialog.TextResult;
                newModelSet.ModelTemplate.Name = textInputDialog.TextResult;
                foreach (var item in newModelSet.ModelFiles)
                {
                    item.OnnxModelPath = item.Type == OnnxModelType.Tokenizer ? _defaultTokenizerPath : null;
                    item.IsOverrideEnabled = false;
                }

                ModelSets.Add(newModelSet);
                SelectedModelSet = newModelSet;
            }
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can execute copy.
        /// </summary>
        /// <returns>
        ///   <c>true</c> if this instance can execute copy; otherwise, <c>false</c>.
        /// </returns>
        private bool CanExecuteCopy()
        {
            return SelectedModelSet?.IsInstalled ?? false;
        }


        /// <summary>
        /// Removes the modelset.
        /// </summary>
        private async Task Remove()
        {
            var textInputDialog = _dialogService.GetDialog<MessageDialog>();
            if (textInputDialog.ShowDialog("Remove ModelSet", "Are you sure you want to remove this ModelSet?", MessageDialog.MessageDialogType.YesNo))
            {
                // Unload and Remove ModelSet
                await UnloadAndRemoveModelSetAsync(SelectedModelSet.Name);

                // Update Template if one was reomved
                UpdateTemplateStatus(SelectedModelSet.Name, ModelTemplateStatus.Deleted);

                // Save Config File
                await SaveConfigurationFile();

                // Remove from edit list
                ModelSets.Remove(SelectedModelSet);
                SelectedModelSet = ModelSets.FirstOrDefault();

                // Force Rebind
                ModelOptions = new ObservableCollection<ModelOptionsModel>(ModelOptions);
            }
        }


        /// <summary>
        /// Determines whether this instance can execute remove.
        /// </summary>
        /// <returns>
        ///   <c>true</c> if this instance can execute remove; otherwise, <c>false</c>.
        /// </returns>
        private bool CanExecuteRemove()
        {
            return SelectedModelSet?.IsDownloading == false;
        }


        /// <summary>
        /// Renames the model.
        /// </summary>
        private async Task Rename()
        {
            var invalidNames = ModelSets.Select(x => x.Name).ToList();
            var textInputDialog = _dialogService.GetDialog<TextInputDialog>();
            if (textInputDialog.ShowDialog("Rename Model Set", "New Name", 1, 30, invalidNames))
            {
                await UnloadAndRemoveModelSetAsync(SelectedModelSet.Name);
                SelectedModelSet.Name = textInputDialog.TextResult.Trim();
                SelectedModelSet.ModelTemplate.Name = textInputDialog.TextResult.Trim();
                await SaveModelAsync(SelectedModelSet);
            }
        }

        #endregion

        #region Import/Export


        /// <summary>
        /// Imports a model template.
        /// </summary>
        private async Task Import()
        {
            try
            {
                var openFileDialog = new OpenFileDialog
                {
                    Title = "Import Model Template",
                    Filter = "json files (*.json)|*.json",
                    DefaultExt = "png",
                    AddExtension = true,
                    RestoreDirectory = true,
                    InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments)
                };

                var dialogResult = openFileDialog.ShowDialog();
                if (dialogResult == false)
                {
                    _logger.LogInformation("Import template canceled");
                    return;
                }

                var serializerOptions = new JsonSerializerOptions();
                serializerOptions.Converters.Add(new JsonStringEnumConverter());
                using var filestream = File.OpenRead(openFileDialog.FileName);
                var modelTemplate = await JsonSerializer.DeserializeAsync<ModelConfigTemplate>(filestream, serializerOptions);
                if (modelTemplate is null)
                {
                    _logger.LogError("Imported template was null");
                    return;
                }

                if (ModelSets.Any(x => x.Name == modelTemplate.Name))
                {
                    _logger.LogError("Imported template already exixts");
                    return;
                }

                // Add
                ModelSets.Add(CreateViewModel(modelTemplate));

                // add to config file and save
                UISettings.ModelTemplates.Add(modelTemplate);
                await SaveConfigurationFile();
            }
            catch (Exception ex)
            {
                _logger.LogError($"Exception importing model template\n{ex}");
            }
        }


        /// <summary>
        /// Exports the model template.
        /// </summary>
        private async Task Export()
        {
            try
            {
                var saveFileDialog = new SaveFileDialog
                {
                    Title = "Export Model Template",
                    Filter = "json files (*.json)|*.json",
                    DefaultExt = "png",
                    AddExtension = true,
                    RestoreDirectory = true,
                    InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                    FileName = $"{SelectedModelSet.Name}.json"
                };

                var dialogResult = saveFileDialog.ShowDialog();
                if (dialogResult == false)
                {
                    _logger.LogInformation("Export template canceled");
                    return;
                }

                // Write file
                var serializerOptions = new JsonSerializerOptions();
                serializerOptions.Converters.Add(new JsonStringEnumConverter());
                serializerOptions.WriteIndented = true;
                using (var appsettingWriteStream = File.Open(saveFileDialog.FileName, FileMode.Create))
                    await JsonSerializer.SerializeAsync(appsettingWriteStream, SelectedModelSet.ModelTemplate, serializerOptions);
            }
            catch (Exception ex)
            {
                _logger.LogError($"Exception exporting model template\n{ex}");
            }
        }


        /// <summary>
        /// Determines whether this instance can execute Export.
        /// </summary>
        /// <returns>
        ///   <c>true</c> if this instance can execute Export; otherwise, <c>false</c>.
        /// </returns>
        private bool CanExecuteExport()
        {
            return SelectedModelSet?.ModelTemplate is not null;
        }

        #endregion

        #region Helper Methods


        /// <summary>
        /// Unloads the and remove model set.
        /// </summary>
        /// <param name="name">The name.</param>
        /// <returns></returns>
        private async Task<bool> UnloadAndRemoveModelSetAsync(string name)
        {
            var onnxModelSet = _stableDiffusionConfig.OnnxModelSets.FirstOrDefault(x => x.Name == name);
            if (onnxModelSet is not null)
            {
                // If model is loaded unload now
                var isLoaded = _stableDiffusionService.IsModelLoaded(onnxModelSet);
                if (isLoaded)
                    await _stableDiffusionService.UnloadModelAsync(onnxModelSet);

                // Remove ViewModel
                var viewModel = ModelOptions.FirstOrDefault(x => x.Name == onnxModelSet.Name);
                if (viewModel is not null)
                    ModelOptions.Remove(viewModel);

                // Remove ModelSet
                _stableDiffusionConfig.OnnxModelSets.Remove(onnxModelSet);
                return true;
            }
            return false;
        }


        /// <summary>
        /// Updates the template status.
        /// </summary>
        /// <param name="name">The name.</param>
        /// <param name="status">The status.</param>
        private void UpdateTemplateStatus(string name, ModelTemplateStatus status)
        {
            // Update Templater if one was used
            var template = UISettings.ModelTemplates.FirstOrDefault(x => x.Name == name);
            if (template is not null)
            {
                if (status == ModelTemplateStatus.Deleted)
                {
                    UISettings.ModelTemplates.Remove(template);
                    return;
                }
                template.Status = status;
            }
        }


        /// <summary>
        /// Validates the model set.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        private bool ValidateModelSet(ModelOptions model)
        {
            if (model == null)
                return false;

            if (!model.ModelConfigurations.Any())
                return false;

            var filesToValidate = model.PipelineType == DiffuserPipelineType.StableDiffusionXL
                ? model.ModelConfigurations
                : model.ModelConfigurations.Where(x => x.Type != OnnxModelType.Tokenizer2 && x.Type != OnnxModelType.TextEncoder2);

            if (filesToValidate.Any(x => !File.Exists(x.OnnxModelPath)))
                return false;

            if (!model.Diffusers.Any())
                return false;

            return true;
        }


        /// <summary>
        /// Saves the configuration file.
        /// </summary>
        /// <returns></returns>
        private Task<bool> SaveConfigurationFile()
        {
            try
            {
                ConfigManager.SaveConfiguration(UISettings);
                ConfigManager.SaveConfiguration(nameof(OnnxStackConfig), _stableDiffusionConfig);
                return Task.FromResult(true);
            }
            catch (Exception ex)
            {
                _logger.LogError($"Error saving configuration file, {ex.Message}");
                return Task.FromResult(false);
            }
        }


        /// <summary>
        /// Creates the view model.
        /// </summary>
        /// <param name="modelTemplate">The model template.</param>
        /// <returns></returns>
        private ModelSetViewModel CreateViewModel(ModelConfigTemplate modelTemplate)
        {
            return new ModelSetViewModel
            {
                IsTemplate = true,
                IsInstalled = false,
                IsEnabled = false,
                Name = modelTemplate.Name,
                BlankTokenId = modelTemplate.BlankTokenId,
                EmbeddingsLength = modelTemplate.EmbeddingsLength,
                ExecutionProvider = ExecutionProvider.Cpu,
                PadTokenId = modelTemplate.PadTokenId,
                ScaleFactor = modelTemplate.ScaleFactor,
                TokenizerLimit = modelTemplate.TokenizerLimit,
                IsDualTokenizer = modelTemplate.IsDualTokenizer,
                SampleSize = modelTemplate.SampleSize,
                DualEmbeddingsLength = modelTemplate.DualEmbeddingsLength,
                PipelineType = modelTemplate.PipelineType,
                EnableTextToImage = modelTemplate.Diffusers.Contains(DiffuserType.TextToImage),
                EnableImageToImage = modelTemplate.Diffusers.Contains(DiffuserType.ImageToImage),
                EnableImageInpaint = modelTemplate.Diffusers.Contains(DiffuserType.ImageInpaint) || modelTemplate.Diffusers.Contains(DiffuserType.ImageInpaintLegacy),
                EnableImageInpaintLegacy = modelTemplate.Diffusers.Contains(DiffuserType.ImageInpaintLegacy),
                ModelFiles = new ObservableCollection<ModelFileViewModel>(Enum.GetValues<OnnxModelType>().Select(x => new ModelFileViewModel { Type = x })),
                ModelTemplate = new ModelConfigTemplate
                {
                    Name = modelTemplate.Name,
                    Author = modelTemplate.Author,
                    BlankTokenId = modelTemplate.BlankTokenId,
                    PadTokenId = modelTemplate.PadTokenId,
                    ScaleFactor = modelTemplate.ScaleFactor,
                    TokenizerLimit = modelTemplate.TokenizerLimit,
                    PipelineType = modelTemplate.PipelineType,
                    Description = modelTemplate.Description,
                    Diffusers = modelTemplate.Diffusers,
                    EmbeddingsLength = modelTemplate.EmbeddingsLength,
                    ImageIcon = modelTemplate.ImageIcon,
                    Images = modelTemplate.Images,
                    ModelFiles = modelTemplate.ModelFiles.ToList(),
                    Repository = modelTemplate.Repository,
                    IsDualTokenizer = modelTemplate.IsDualTokenizer,
                    SampleSize = modelTemplate.SampleSize,
                    DualEmbeddingsLength = modelTemplate.DualEmbeddingsLength,
                    Status = ModelTemplateStatus.Installed
                }
            };
        }


        /// <summary>
        /// Creates the view model.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <returns></returns>
        private ModelSetViewModel CreateViewModel(ModelOptions modelOptions)
        {
            var isValid = ValidateModelSet(modelOptions);
            return new ModelSetViewModel
            {
                IsTemplate = false,
                IsInstalled = isValid,
                IsEnabled = isValid && modelOptions.IsEnabled,
                Name = modelOptions.Name,
                BlankTokenId = modelOptions.BlankTokenId,
                DeviceId = modelOptions.DeviceId,
                EmbeddingsLength = modelOptions.EmbeddingsLength,
                ExecutionMode = modelOptions.ExecutionMode,
                ExecutionProvider = modelOptions.ExecutionProvider,
                IntraOpNumThreads = modelOptions.IntraOpNumThreads,
                InterOpNumThreads = modelOptions.InterOpNumThreads,
                PadTokenId = modelOptions.PadTokenId,
                ScaleFactor = modelOptions.ScaleFactor,
                IsDualTokenizer = modelOptions.IsDualTokenizer,
                SampleSize = modelOptions.SampleSize,
                DualEmbeddingsLength = modelOptions.DualEmbeddingsLength,
                TokenizerLimit = modelOptions.TokenizerLimit,
                PipelineType = modelOptions.PipelineType,
                EnableTextToImage = modelOptions.Diffusers.Contains(DiffuserType.TextToImage),
                EnableImageToImage = modelOptions.Diffusers.Contains(DiffuserType.ImageToImage),
                EnableImageInpaint = modelOptions.Diffusers.Contains(DiffuserType.ImageInpaint) || modelOptions.Diffusers.Contains(DiffuserType.ImageInpaintLegacy),
                EnableImageInpaintLegacy = modelOptions.Diffusers.Contains(DiffuserType.ImageInpaintLegacy),
                ModelFiles = new ObservableCollection<ModelFileViewModel>(modelOptions.ModelConfigurations.Select(x => new ModelFileViewModel
                {
                    Type = x.Type,
                    DeviceId = x.DeviceId ?? modelOptions.DeviceId,
                    ExecutionMode = x.ExecutionMode ?? modelOptions.ExecutionMode,
                    ExecutionProvider = x.ExecutionProvider ?? modelOptions.ExecutionProvider,
                    IntraOpNumThreads = x.IntraOpNumThreads ?? modelOptions.IntraOpNumThreads,
                    InterOpNumThreads = x.InterOpNumThreads ?? modelOptions.InterOpNumThreads,
                    OnnxModelPath = x.OnnxModelPath,
                    IsOverrideEnabled =
                           x.DeviceId.HasValue
                        || x.ExecutionMode.HasValue
                        || x.ExecutionProvider.HasValue
                        || x.IntraOpNumThreads.HasValue
                        || x.InterOpNumThreads.HasValue
                })),
                ModelTemplate = new ModelConfigTemplate
                {
                    Name = modelOptions.Name,
                    Author = "",
                    BlankTokenId = modelOptions.BlankTokenId,
                    PadTokenId = modelOptions.PadTokenId,
                    ScaleFactor = modelOptions.ScaleFactor,
                    TokenizerLimit = modelOptions.TokenizerLimit,
                    PipelineType = modelOptions.PipelineType,
                    IsDualTokenizer = modelOptions.IsDualTokenizer,
                    SampleSize = modelOptions.SampleSize,
                    DualEmbeddingsLength = modelOptions.DualEmbeddingsLength,
                    Description = "",
                    Diffusers = modelOptions.Diffusers,
                    EmbeddingsLength = modelOptions.EmbeddingsLength,
                    ImageIcon = "",
                    Images = Enumerable.Range(0, 6).Select(x => string.Empty).ToList(),
                    ModelFiles = new List<string>(),
                    Repository = "",
                    Status = ModelTemplateStatus.Installed
                }
            };
        }


        /// <summary>
        /// Creates the model options.
        /// </summary>
        /// <param name="editModel">The edit model.</param>
        /// <returns></returns>
        private ModelOptions CreateModelOptions(ModelSetViewModel editModel)
        {
            return new ModelOptions
            {
                IsEnabled = editModel.IsEnabled,
                Name = editModel.Name,
                BlankTokenId = editModel.BlankTokenId,
                DeviceId = editModel.DeviceId,
                EmbeddingsLength = editModel.EmbeddingsLength,
                ExecutionMode = editModel.ExecutionMode,
                ExecutionProvider = editModel.ExecutionProvider,
                IntraOpNumThreads = editModel.IntraOpNumThreads,
                InterOpNumThreads = editModel.InterOpNumThreads,
                PadTokenId = editModel.PadTokenId,
                ScaleFactor = editModel.ScaleFactor,
                TokenizerLimit = editModel.TokenizerLimit,
                PipelineType = editModel.PipelineType,
                Diffusers = new List<DiffuserType>(editModel.GetDiffusers()),
                DualEmbeddingsLength = editModel.DualEmbeddingsLength,
                SampleSize = editModel.SampleSize,
                IsDualTokenizer = editModel.IsDualTokenizer,
                ModelConfigurations = new List<OnnxModelSessionConfig>(editModel.ModelFiles.Select(x => new OnnxModelSessionConfig
                {
                    Type = x.Type,
                    OnnxModelPath = x.OnnxModelPath,
                    DeviceId = x.IsOverrideEnabled ? x.DeviceId : default,
                    ExecutionMode = x.IsOverrideEnabled ? x.ExecutionMode : default,
                    ExecutionProvider = x.IsOverrideEnabled ? x.ExecutionProvider : default,
                    IntraOpNumThreads = x.IsOverrideEnabled ? x.IntraOpNumThreads : default,
                    InterOpNumThreads = x.IsOverrideEnabled ? x.InterOpNumThreads : default
                }))
            };
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
