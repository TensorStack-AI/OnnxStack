using Microsoft.ML.OnnxRuntime;
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
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace OnnxStack.UI.Views
{
    /// <summary>
    /// Interaction logic for Settings.xaml
    /// </summary>
    public partial class Settings : UserControl, INavigatable, INotifyPropertyChanged
    {
        private readonly StableDiffusionConfig _stableDiffusionConfig;
        private readonly IDialogService _dialogService;
        private readonly IOnnxModelService _onnxModelService;
        private readonly IStableDiffusionService _stableDiffusionService;
        private ModelOptionsModel _selectedModel;
        private ModelSetEditModel _selectedModelSet;
        private ObservableCollection<ModelSetEditModel> _modelSets;

        public Settings()
        {
            if (!DesignerProperties.GetIsInDesignMode(this))
            {
                _dialogService = App.GetService<IDialogService>();
                _onnxModelService = App.GetService<IOnnxModelService>();
                _stableDiffusionConfig = App.GetService<StableDiffusionConfig>();
                _stableDiffusionService = App.GetService<IStableDiffusionService>();
            }

            SaveCommand = new AsyncRelayCommand(Save, CanExecuteSave);
            ResetCommand = new AsyncRelayCommand(Reset, CanExecuteReset);
            AddCommand = new AsyncRelayCommand(Add, CanExecuteAdd);
            CopyCommand = new AsyncRelayCommand(Copy, CanExecuteCopy);
            RemoveCommand = new AsyncRelayCommand(Remove, CanExecuteRemove);

            Initialize();
            InitializeComponent();
        }

        public AsyncRelayCommand SaveCommand { get; }
        public AsyncRelayCommand ResetCommand { get; }
        public AsyncRelayCommand AddCommand { get; }
        public AsyncRelayCommand CopyCommand { get; }
        public AsyncRelayCommand RemoveCommand { get; }

        public ObservableCollection<ModelOptionsModel> ModelOptions
        {
            get { return (ObservableCollection<ModelOptionsModel>)GetValue(ModelOptionsProperty); }
            set { SetValue(ModelOptionsProperty, value); }
        }

        public static readonly DependencyProperty ModelOptionsProperty =
            DependencyProperty.Register("ModelOptions", typeof(ObservableCollection<ModelOptionsModel>), typeof(Settings));

        public ModelOptionsModel SelectedModelOption
        {
            get { return (ModelOptionsModel)GetValue(SelectedModelOptionProperty); }
            set { SetValue(SelectedModelOptionProperty, value); }
        }

        public static readonly DependencyProperty SelectedModelOptionProperty =
            DependencyProperty.Register("SelectedModelOption", typeof(ModelOptionsModel), typeof(Settings));


        public ObservableCollection<ModelSetEditModel> ModelSets
        {
            get { return _modelSets; }
            set { _modelSets = value; }
        }

        public ModelSetEditModel SelectedModelSet
        {
            get { return _selectedModelSet; }
            set { _selectedModelSet = value; NotifyPropertyChanged(); }
        }

        public Task NavigateAsync(ImageResult imageResult)
        {
            return Task.CompletedTask;
        }

        private async Task<bool> UnloadAndRemoveModelSet(string name)
        {
            var modelSet = _stableDiffusionConfig.OnnxModelSets.FirstOrDefault(x => x.Name == name);
            if (modelSet is not null)
            {
                // If model is loaded unload now
                var isLoaded = _stableDiffusionService.IsModelLoaded(modelSet);
                if (isLoaded)
                    await _stableDiffusionService.UnloadModel(modelSet);

                // Remove ViewModel
                var viewModel = ModelOptions.FirstOrDefault(x => x.Name == modelSet.Name);
                if (viewModel is not null)
                    ModelOptions.Remove(viewModel);

                // Remove ModelSet
                _stableDiffusionConfig.OnnxModelSets.Remove(modelSet);
                return true;
            }
            return false;
        }

        private async Task Save()
        {
            // Unload and Remove ModelSet
            await UnloadAndRemoveModelSet(SelectedModelSet.Name);

            // Create New ModelSet
            var newModelOption = CreateModelOptions(SelectedModelSet);
            newModelOption.InitBlankTokenArray();


            // Add to Config file
            _stableDiffusionConfig.OnnxModelSets.Add(newModelOption);

            // Save Config File
            if (!await SaveConfigurationFile(_stableDiffusionConfig))
            {
                // LOG ME
                return;
            }

            // Update OnnxStack Service
            newModelOption.ApplyConfigurationOverrides();
            _onnxModelService.UpdateModelSet(newModelOption);

            // Add new ViewModel
            ModelOptions.Add(new ModelOptionsModel
            {
                Name = SelectedModelSet.Name,
                ModelOptions = newModelOption,
                IsEnabled = newModelOption.IsEnabled
            });
            ModelOptions = new ObservableCollection<ModelOptionsModel>(ModelOptions);
        }


        private bool CanExecuteSave()
        {
            return true;
        }


        private Task Reset()
        {
            Initialize();
            return Task.CompletedTask;
        }

        private bool CanExecuteReset()
        {
            return true;
        }

        private Task Add()
        {
            var invalidNames = ModelSets.Select(x => x.Name).ToList();
            var textInputDialog = _dialogService.GetDialog<TextInputDialog>();
            if (textInputDialog.ShowDialog("Add Model Set", "Name", 1, 30, invalidNames))
            {
                var models = Enum.GetValues<OnnxModelType>()
                    .Where(x => x != OnnxModelType.SafetyChecker)
                    .Select(x => new ModelSessionEditModel { Type = x });
                var newModelSet = new ModelSetEditModel
                {
                    Name = textInputDialog.TextResult,
                    ModelConfigurations = new ObservableCollection<ModelSessionEditModel>(models)
                };
                ModelSets.Add(newModelSet);
                SelectedModelSet = newModelSet;
            }
            return Task.CompletedTask;
        }

        private bool CanExecuteAdd()
        {
            return true;
        }

        private Task Copy()
        {
            var invalidNames = ModelSets.Select(x => x.Name).ToList();
            var textInputDialog = _dialogService.GetDialog<TextInputDialog>();
            if (textInputDialog.ShowDialog("Copy Model Set", "New Name", 1, 30, invalidNames))
            {
                var original = _stableDiffusionConfig.OnnxModelSets.FirstOrDefault(x => x.Name == SelectedModelSet.Name);
                var newModelSet = CreateEditModel(original);
                newModelSet.Name = textInputDialog.TextResult;
                ModelSets.Add(newModelSet);
                SelectedModelSet = newModelSet;
            }
            return Task.CompletedTask;
        }


        private bool CanExecuteCopy()
        {
            return _stableDiffusionConfig.OnnxModelSets.Any(x => x.Name == SelectedModelSet?.Name);
        }


        private async Task Remove()
        {
            var textInputDialog = _dialogService.GetDialog<MessageDialog>();
            if (textInputDialog.ShowDialog("Remove ModelSet", "Are you sure you want to remove this ModelSet?", MessageDialog.MessageDialogType.YesNo))
            {
                // Unload and Remove ModelSet
                if (await UnloadAndRemoveModelSet(SelectedModelSet.Name))
                {
                    if (!SaveConfigurationFile(_stableDiffusionConfig).Result)
                    {
                        // LOG ME
                    }
                }

                // Remove from edit list
                ModelSets.Remove(SelectedModelSet);
                SelectedModelSet = ModelSets.FirstOrDefault();

                // Notify ViewModel
                ModelOptions = new ObservableCollection<ModelOptionsModel>(ModelOptions);
            }
        }

        private bool CanExecuteRemove()
        {
            return true;
        }


        private void Initialize()
        {
            ModelSets = new ObservableCollection<ModelSetEditModel>(_stableDiffusionConfig.OnnxModelSets.Select(CreateEditModel));
        }

        private ModelSetEditModel CreateEditModel(ModelOptions modelOptions)
        {
            return new ModelSetEditModel
            {
                Name = modelOptions.Name,
                BlankTokenId = modelOptions.BlankTokenId,
                DeviceId = modelOptions.DeviceId,
                EmbeddingsLength = modelOptions.EmbeddingsLength,
                ExecutionMode = modelOptions.ExecutionMode,
                ExecutionProvider = modelOptions.ExecutionProvider,
                IntraOpNumThreads = modelOptions.IntraOpNumThreads,
                InterOpNumThreads = modelOptions.InterOpNumThreads,
                InputTokenLimit = modelOptions.InputTokenLimit,
                IsEnabled = modelOptions.IsEnabled,
                PadTokenId = modelOptions.PadTokenId,
                ScaleFactor = modelOptions.ScaleFactor,
                TokenizerLimit = modelOptions.TokenizerLimit,
                PipelineType = modelOptions.PipelineType,
                EnableTextToImage = modelOptions.Diffusers.Contains(DiffuserType.TextToImage),
                EnableImageToImage = modelOptions.Diffusers.Contains(DiffuserType.ImageToImage),
                EnableImageInpaint = modelOptions.Diffusers.Contains(DiffuserType.ImageInpaint) || modelOptions.Diffusers.Contains(DiffuserType.ImageInpaintLegacy),
                EnableImageInpaintLegacy = modelOptions.Diffusers.Contains(DiffuserType.ImageInpaintLegacy),
                ModelConfigurations = new ObservableCollection<ModelSessionEditModel>(modelOptions.ModelConfigurations.Select(x => new ModelSessionEditModel
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
                }))
            };
        }


        private ModelOptions CreateModelOptions(ModelSetEditModel editModel)
        {
            return new ModelOptions
            {
                Name = editModel.Name,
                BlankTokenId = editModel.BlankTokenId,
                DeviceId = editModel.DeviceId,
                EmbeddingsLength = editModel.EmbeddingsLength,
                ExecutionMode = editModel.ExecutionMode,
                ExecutionProvider = editModel.ExecutionProvider,
                IntraOpNumThreads = editModel.IntraOpNumThreads,
                InterOpNumThreads = editModel.InterOpNumThreads,
                InputTokenLimit = editModel.InputTokenLimit,
                IsEnabled = editModel.IsEnabled,
                PadTokenId = editModel.PadTokenId,
                ScaleFactor = editModel.ScaleFactor,
                TokenizerLimit = editModel.TokenizerLimit,
                PipelineType = editModel.PipelineType,
                Diffusers = new List<DiffuserType>(editModel.GetDiffusers()),
                ModelConfigurations = new List<OnnxModelSessionConfig>(editModel.ModelConfigurations.Select(x => new OnnxModelSessionConfig
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


        private Task<bool> SaveConfigurationFile(StableDiffusionConfig stableDiffusionConfig)
        {
            try
            {
                ConfigManager.SaveConfiguration(nameof(OnnxStackConfig), stableDiffusionConfig);
                return Task.FromResult(true);
            }
            catch (Exception ex)
            {
                // LOG ME
                return Task.FromResult(false);
            }
        }

        #region INotifyPropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;
        public void NotifyPropertyChanged([CallerMemberName] string property = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(property));
        }
        #endregion
    }


    public class ModelSetEditModel : INotifyPropertyChanged
    {
        private string _name;
        private bool _isEnabled;
        private int _deviceId;
        private int _interOpNumThreads;
        private int _intraOpNumThreads;
        private ExecutionMode _executionMode;
        private ExecutionProvider _executionProvider;
        private ObservableCollection<ModelSessionEditModel> _modelConfigurations;
        private int _padTokenId;
        private int _blankTokenId;
        private float _scaleFactor;
        private int _tokenizerLimit;
        private int _inputTokenLimit;
        private int _embeddingsLength;
        private bool _enableTextToImage;
        private bool _enableImageToImage;
        private bool _enableImageInpaint;
        private bool _enableImageInpaintLegacy;
        private DiffuserPipelineType _pipelineType;

        public string Name
        {
            get { return _name; }
            set { _name = value; NotifyPropertyChanged(); }
        }

        public bool IsEnabled
        {
            get { return _isEnabled; }
            set { _isEnabled = value; NotifyPropertyChanged(); }
        }

        public int PadTokenId
        {
            get { return _padTokenId; }
            set { _padTokenId = value; NotifyPropertyChanged(); }
        }
        public int BlankTokenId
        {
            get { return _blankTokenId; }
            set { _blankTokenId = value; NotifyPropertyChanged(); }
        }
        public float ScaleFactor
        {
            get { return _scaleFactor; }
            set { _scaleFactor = value; NotifyPropertyChanged(); }
        }
        public int TokenizerLimit
        {
            get { return _tokenizerLimit; }
            set { _tokenizerLimit = value; NotifyPropertyChanged(); }
        }
        public int InputTokenLimit
        {
            get { return _inputTokenLimit; }
            set { _inputTokenLimit = value; NotifyPropertyChanged(); }
        }
        public int EmbeddingsLength
        {
            get { return _embeddingsLength; }
            set { _embeddingsLength = value; NotifyPropertyChanged(); }
        }

        public bool EnableTextToImage
        {
            get { return _enableTextToImage; }
            set { _enableTextToImage = value; NotifyPropertyChanged(); }
        }

        public bool EnableImageToImage
        {
            get { return _enableImageToImage; }
            set { _enableImageToImage = value; NotifyPropertyChanged(); }
        }

        public bool EnableImageInpaint
        {
            get { return _enableImageInpaint; }
            set
            {
                _enableImageInpaint = value;
                NotifyPropertyChanged();

                if (!_enableImageInpaint)
                    EnableImageInpaintLegacy = false;
            }
        }

        public bool EnableImageInpaintLegacy
        {
            get { return _enableImageInpaintLegacy; }
            set
            {
                _enableImageInpaintLegacy = value;
                NotifyPropertyChanged();
            }
        }

        public int DeviceId
        {
            get { return _deviceId; }
            set { _deviceId = value; NotifyPropertyChanged(); }
        }

        public int InterOpNumThreads
        {
            get { return _interOpNumThreads; }
            set { _interOpNumThreads = value; NotifyPropertyChanged(); }
        }

        public int IntraOpNumThreads
        {
            get { return _intraOpNumThreads; }
            set { _intraOpNumThreads = value; NotifyPropertyChanged(); }
        }

        public ExecutionMode ExecutionMode
        {
            get { return _executionMode; }
            set { _executionMode = value; NotifyPropertyChanged(); }
        }

        public ExecutionProvider ExecutionProvider
        {
            get { return _executionProvider; }
            set { _executionProvider = value; NotifyPropertyChanged(); }
        }

        public ObservableCollection<ModelSessionEditModel> ModelConfigurations
        {
            get { return _modelConfigurations; }
            set { _modelConfigurations = value; NotifyPropertyChanged(); }
        }

        public DiffuserPipelineType PipelineType
        {
            get { return _pipelineType; }
            set { _pipelineType = value; NotifyPropertyChanged(); }
        }


        public IEnumerable<DiffuserType> GetDiffusers()
        {
            if (_enableTextToImage)
                yield return DiffuserType.TextToImage;
            if (_enableImageToImage)
                yield return DiffuserType.ImageToImage;
            if (_enableImageInpaint && !_enableImageInpaintLegacy)
                yield return DiffuserType.ImageInpaint;
            if (_enableImageInpaint && _enableImageInpaintLegacy)
                yield return DiffuserType.ImageInpaintLegacy;
        }

        #region INotifyPropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;
        public void NotifyPropertyChanged([CallerMemberName] string property = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(property));
        }
        #endregion
    }

    public class ModelSessionEditModel : INotifyPropertyChanged
    {
        private OnnxModelType _type;
        private string _onnnxModelPath;
        private bool? _isEnabled;
        private int? _deviceId;
        private int? _interOpNumThreads;
        private int? _intraOpNumThreads;
        private ExecutionMode? _executionMode;
        private ExecutionProvider? _executionProvider;

        public string OnnxModelPath
        {
            get { return _onnnxModelPath; }
            set { _onnnxModelPath = value; NotifyPropertyChanged(); }
        }
        public bool? IsEnabled
        {
            get { return _isEnabled; }
            set { _isEnabled = value; NotifyPropertyChanged(); }
        }
        public int? DeviceId
        {
            get { return _deviceId; }
            set { _deviceId = value; NotifyPropertyChanged(); }
        }
        public int? InterOpNumThreads
        {
            get { return _interOpNumThreads; }
            set { _interOpNumThreads = value; NotifyPropertyChanged(); }
        }
        public int? IntraOpNumThreads
        {
            get { return _intraOpNumThreads; }
            set { _intraOpNumThreads = value; NotifyPropertyChanged(); }
        }
        public ExecutionMode? ExecutionMode
        {
            get { return _executionMode; }
            set { _executionMode = value; NotifyPropertyChanged(); }
        }
        public ExecutionProvider? ExecutionProvider
        {
            get { return _executionProvider; }
            set { _executionProvider = value; NotifyPropertyChanged(); }
        }

        public OnnxModelType Type
        {
            get { return _type; }
            set { _type = value; NotifyPropertyChanged(); }
        }


        private bool _isOverrideEnabled;

        public bool IsOverrideEnabled
        {
            get { return _isOverrideEnabled; }
            set { _isOverrideEnabled = value; NotifyPropertyChanged(); }
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
