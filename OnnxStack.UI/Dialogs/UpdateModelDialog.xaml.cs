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
    /// Interaction logic for UpdateModelDialog.xaml
    /// </summary>
    public partial class UpdateModelDialog : Window, INotifyPropertyChanged
    {
        private readonly ILogger<UpdateModelDialog> _logger;

        private List<string> _invalidOptions;
        private DiffuserPipelineType _pipelineType;
        private ModelType _modelType;
        private string _modelFolder;
        private string _modelName;
        private string _defaultTokenizerPath;
        private OnnxStackUIConfig _uiSettings;

        public UpdateModelDialog(OnnxStackUIConfig uiSettings, ILogger<UpdateModelDialog> logger)
        {
            var defaultTokenizerPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "cliptokenizer.onnx");
            if (File.Exists(defaultTokenizerPath))
                _defaultTokenizerPath = defaultTokenizerPath;

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

        public DiffuserPipelineType PipelineType
        {
            get { return _pipelineType; }
            set
            {
                _pipelineType = value;
                NotifyPropertyChanged();
                if (_pipelineType != DiffuserPipelineType.StableDiffusionXL && _pipelineType != DiffuserPipelineType.LatentConsistencyXL)
                {
                    _modelType = ModelType.Base;
                    NotifyPropertyChanged(nameof(ModelType));
                }
                CreateModelSet();
            }
        }


        public ModelType ModelType
        {
            get { return _modelType; }
            set { _modelType = value; NotifyPropertyChanged(); CreateModelSet(); }
        }

        public string ModelName
        {
            get { return _modelName; }
            set { _modelName = value; NotifyPropertyChanged(); CreateModelSet(); }
        }


        public string ModelFolder
        {
            get { return _modelFolder; }
            set
            {
                _modelFolder = value;
                _modelName = string.IsNullOrEmpty(_modelFolder)
                    ? string.Empty
                    : Path.GetFileName(_modelFolder);

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


        private StableDiffusionModelSet _modelSet;

        public StableDiffusionModelSet ModelSet
        {
            get { return _modelSet; }
            set { _modelSet = value; NotifyPropertyChanged(); }
        }



        private void CreateModelSet()
        {
            ModelSet = null;
            IsNameInvalid = false;
            ValidationResults.Clear();
            if (string.IsNullOrEmpty(_modelFolder))
                return;

            ModelSet = new StableDiffusionModelSet
            {
                Name = ModelName.Trim(),
                PipelineType = PipelineType,
                ScaleFactor = 0.18215f,
                TokenizerLimit = 77,
                PadTokenId = 49407,
                TokenizerLength = 768,
                Tokenizer2Length = 1280,
                BlankTokenId = 49407,
                Diffusers = Enum.GetValues<DiffuserType>().ToList(),
                SampleSize = 512,
                TokenizerType = TokenizerType.One,
                ModelType = ModelType.Base,

                DeviceId = _uiSettings.DefaultDeviceId,
                ExecutionMode = _uiSettings.DefaultExecutionMode,
                ExecutionProvider = _uiSettings.DefaultExecutionProvider,
                InterOpNumThreads = _uiSettings.DefaultInterOpNumThreads,
                IntraOpNumThreads = _uiSettings.DefaultIntraOpNumThreads,
                IsEnabled = true,
                ModelConfigurations = new List<OnnxModelConfig>()
            };


            var unetPath = Path.Combine(ModelFolder, "unet", "model.onnx");
            var tokenizerPath = Path.Combine(ModelFolder, "tokenizer", "model.onnx");
            var textEncoderPath = Path.Combine(ModelFolder, "text_encoder", "model.onnx");
            var vaeDecoder = Path.Combine(ModelFolder, "vae_decoder", "model.onnx");
            var vaeEncoder = Path.Combine(ModelFolder, "vae_encoder", "model.onnx");
            var tokenizer2Path = Path.Combine(ModelFolder, "tokenizer_2", "model.onnx");
            var textEncoder2Path = Path.Combine(ModelFolder, "text_encoder_2", "model.onnx");
            if (!File.Exists(tokenizerPath))
                tokenizerPath = _defaultTokenizerPath;
            if (!File.Exists(tokenizer2Path))
                tokenizer2Path = _defaultTokenizerPath;

            if (PipelineType == DiffuserPipelineType.StableDiffusionXL || PipelineType == DiffuserPipelineType.LatentConsistencyXL)
            {
                ModelSet.PadTokenId = 1;
                ModelSet.SampleSize = 1024;
                ModelSet.ScaleFactor = 0.13025f;
                ModelSet.TokenizerType = TokenizerType.Both;

                if (ModelType == ModelType.Refiner)
                {
                    ModelSet.ModelType = ModelType.Refiner;
                    ModelSet.TokenizerType = TokenizerType.Two;
                    ModelSet.Diffusers.Remove(DiffuserType.TextToImage);
                    ModelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.Unet, OnnxModelPath = unetPath });
                    ModelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.Tokenizer2, OnnxModelPath = tokenizer2Path });
                    ModelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.TextEncoder2, OnnxModelPath = textEncoder2Path });
                    ModelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.VaeDecoder, OnnxModelPath = vaeDecoder });
                    ModelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.VaeEncoder, OnnxModelPath = vaeEncoder });
                }
                else
                {
                    ModelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.Unet, OnnxModelPath = unetPath });
                    ModelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.Tokenizer, OnnxModelPath = tokenizerPath });
                    ModelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.Tokenizer2, OnnxModelPath = tokenizer2Path });
                    ModelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.TextEncoder, OnnxModelPath = textEncoderPath });
                    ModelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.TextEncoder2, OnnxModelPath = textEncoder2Path });
                    ModelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.VaeDecoder, OnnxModelPath = vaeDecoder });
                    ModelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.VaeEncoder, OnnxModelPath = vaeEncoder });
                }
            }
            else
            {
                ModelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.Unet, OnnxModelPath = unetPath });
                ModelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.Tokenizer, OnnxModelPath = tokenizerPath });
                ModelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.TextEncoder, OnnxModelPath = textEncoderPath });
                ModelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.VaeDecoder, OnnxModelPath = vaeDecoder });
                ModelSet.ModelConfigurations.Add(new OnnxModelConfig { Type = OnnxModelType.VaeEncoder, OnnxModelPath = vaeEncoder });
            }

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


        public bool ShowDialog(StableDiffusionModelSet modelSet, List<string> invalidOptions = null)
        {
            ModelSet = modelSet with { };
            InvalidOptions = invalidOptions;
            return ShowDialog() ?? false;
        }


        private Task Save()
        {
            DialogResult = true;
            return Task.CompletedTask;
        }

        private bool CanExecuteSave()
        {
            if (string.IsNullOrEmpty(_modelFolder))
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
