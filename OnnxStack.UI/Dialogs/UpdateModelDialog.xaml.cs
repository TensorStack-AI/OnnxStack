using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
using OnnxStack.UI.Views;
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
            SaveCommand = new AsyncRelayCommand(Save, CanExecuteSave);
            CancelCommand = new AsyncRelayCommand(Cancel, CanExecuteCancel);
            _invalidOptions = _uiSettings.StableDiffusionModelSets
                .Select(x => x.Name)
                .ToList();
            InitializeComponent();
        }

        public OnnxStackUIConfig UISettings => _uiSettings;
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

        #region INotifyPropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;
        public void NotifyPropertyChanged([CallerMemberName] string property = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(property));
        }
        #endregion
    }

    public class UpdateModelSetViewModel : INotifyPropertyChanged
    {
        private string _name;
        private int _deviceId;
        private int _interOpNumThreads;
        private int _intraOpNumThreads;
        private ExecutionMode _executionMode;
        private ExecutionProvider _executionProvider;
        private ObservableCollection<ModelFileViewModel> _modelFiles;
        private int _padTokenId;
        private int _blankTokenId;
        private float _scaleFactor;
        private int _tokenizerLimit;
        private int _embeddingsLength;
        private bool _enableTextToImage;
        private bool _enableImageToImage;
        private bool _enableImageInpaint;
        private bool _enableImageInpaintLegacy;
        private DiffuserPipelineType _pipelineType;
        private int _dualEmbeddingsLength;
        private TokenizerType _tokenizerType;
        private int _sampleSize;
        private ModelType _modelType;

        public string Name
        {
            get { return _name; }
            set { _name = value; NotifyPropertyChanged(); }
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

        public int SampleSize
        {
            get { return _sampleSize; }
            set { _sampleSize = value; NotifyPropertyChanged(); }
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

        public TokenizerType TokenizerType
        {
            get { return _tokenizerType; }
            set { _tokenizerType = value; NotifyPropertyChanged(); }
        }

        public int Tokenizer2Length
        {
            get { return _dualEmbeddingsLength; }
            set { _dualEmbeddingsLength = value; NotifyPropertyChanged(); }
        }

        public int TokenizerLength
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
            { _enableImageInpaint = value; NotifyPropertyChanged(); }
        }

        public bool EnableImageInpaintLegacy
        {
            get { return _enableImageInpaintLegacy; }
            set { _enableImageInpaintLegacy = value; NotifyPropertyChanged(); }
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

        public ObservableCollection<ModelFileViewModel> ModelFiles
        {
            get { return _modelFiles; }
            set { _modelFiles = value; NotifyPropertyChanged(); }
        }

        public DiffuserPipelineType PipelineType
        {
            get { return _pipelineType; }
            set { _pipelineType = value; NotifyPropertyChanged(); }
        }



        public ModelType ModelType
        {
            get { return _modelType; }
            set { _modelType = value; NotifyPropertyChanged(); }
        }

        public IEnumerable<DiffuserType> GetDiffusers()
        {
            if (_enableTextToImage)
                yield return DiffuserType.TextToImage;
            if (_enableImageToImage)
                yield return DiffuserType.ImageToImage;
            if (_enableImageInpaint)
                yield return DiffuserType.ImageInpaint;
            if (_enableImageInpaintLegacy)
                yield return DiffuserType.ImageInpaintLegacy;
        }



        public static UpdateModelSetViewModel FromModelSet(StableDiffusionModelSet modelset)
        {
            return new UpdateModelSetViewModel
            {
                BlankTokenId = modelset.BlankTokenId,
                DeviceId = modelset.DeviceId,
                EnableImageInpaint = modelset.Diffusers.Contains(DiffuserType.ImageInpaint),
                EnableImageInpaintLegacy = modelset.Diffusers.Contains(DiffuserType.ImageInpaintLegacy),
                EnableImageToImage = modelset.Diffusers.Contains(DiffuserType.ImageToImage),
                EnableTextToImage = modelset.Diffusers.Contains(DiffuserType.TextToImage),
                ExecutionMode = modelset.ExecutionMode,
                ExecutionProvider = modelset.ExecutionProvider,
                InterOpNumThreads = modelset.InterOpNumThreads,
                IntraOpNumThreads = modelset.IntraOpNumThreads,
                ModelType = modelset.ModelType,
                Name = modelset.Name,
                PadTokenId = modelset.PadTokenId,
                PipelineType = modelset.PipelineType,
                SampleSize = modelset.SampleSize,
                ScaleFactor = modelset.ScaleFactor,
                Tokenizer2Length = modelset.Tokenizer2Length,
                TokenizerLength = modelset.TokenizerLength,
                TokenizerLimit = modelset.TokenizerLimit,
                TokenizerType = modelset.TokenizerType,
                ModelFiles = new ObservableCollection<ModelFileViewModel>(modelset.ModelConfigurations.Select(c => new ModelFileViewModel
                {
                    Type = c.Type,
                    OnnxModelPath = c.OnnxModelPath,

                    DeviceId = c.DeviceId ?? modelset.DeviceId,
                    ExecutionMode = c.ExecutionMode ?? modelset.ExecutionMode,
                    ExecutionProvider = c.ExecutionProvider ?? modelset.ExecutionProvider,
                    InterOpNumThreads = c.InterOpNumThreads ?? modelset.InterOpNumThreads,
                    IntraOpNumThreads = c.IntraOpNumThreads ?? modelset.IntraOpNumThreads,
                    IsOverrideEnabled =
                           c.DeviceId.HasValue
                        || c.ExecutionMode.HasValue
                        || c.ExecutionProvider.HasValue
                        || c.IntraOpNumThreads.HasValue
                        || c.InterOpNumThreads.HasValue
                }))
            };
        }

        public static StableDiffusionModelSet ToModelSet(UpdateModelSetViewModel modelset)
        {
            return new StableDiffusionModelSet
            {
                IsEnabled = true,
                Name = modelset.Name,
                PipelineType = modelset.PipelineType,
                ModelType = modelset.ModelType,
                SampleSize = modelset.SampleSize,
                Diffusers = new List<DiffuserType>(modelset.GetDiffusers()),

                DeviceId = modelset.DeviceId,
                ExecutionMode = modelset.ExecutionMode,
                ExecutionProvider = modelset.ExecutionProvider,
                InterOpNumThreads = modelset.InterOpNumThreads,
                IntraOpNumThreads = modelset.IntraOpNumThreads,

                PadTokenId = modelset.PadTokenId,
                BlankTokenId = modelset.BlankTokenId,
                ScaleFactor = modelset.ScaleFactor,
                TokenizerType = modelset.TokenizerType,
                TokenizerLimit = modelset.TokenizerLimit,
                TokenizerLength = modelset.TokenizerLength,
                Tokenizer2Length = modelset.Tokenizer2Length,
                ModelConfigurations = new List<OnnxModelConfig>(modelset.ModelFiles.Select(x => new OnnxModelConfig
                {
                    Type = x.Type,
                    OnnxModelPath = x.OnnxModelPath,
                    DeviceId = x.IsOverrideEnabled && modelset.DeviceId != x.DeviceId ? x.DeviceId : default,
                    ExecutionMode = x.IsOverrideEnabled && modelset.ExecutionMode != x.ExecutionMode ? x.ExecutionMode : default,
                    ExecutionProvider = x.IsOverrideEnabled && modelset.ExecutionProvider != x.ExecutionProvider ? x.ExecutionProvider : default,
                    IntraOpNumThreads = x.IsOverrideEnabled && modelset.IntraOpNumThreads != x.IntraOpNumThreads ? x.IntraOpNumThreads : default,
                    InterOpNumThreads = x.IsOverrideEnabled && modelset.InterOpNumThreads != x.InterOpNumThreads ? x.InterOpNumThreads : default,
                }))
            };
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
