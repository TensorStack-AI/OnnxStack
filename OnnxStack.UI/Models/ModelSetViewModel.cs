using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Enums;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace OnnxStack.UI.Views
{
    public class ModelSetViewModel : INotifyPropertyChanged
    {
        private string _name;
        private bool _isEnabled;
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
        private bool _isInstalled;
        private bool _isTemplate;
        private string _progessText;
        private double _progressValue;
        private bool _isDownloading;

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

        public bool IsInstalled
        {
            get { return _isInstalled; }
            set { _isInstalled = value; NotifyPropertyChanged(); }
        }

        public string ProgessText
        {
            get { return _progessText; }
            set { _progessText = value; NotifyPropertyChanged(); }
        }

        public double ProgressValue
        {
            get { return _progressValue; }
            set { _progressValue = value; NotifyPropertyChanged(); }
        }

        public bool IsDownloading
        {
            get { return _isDownloading; }
            set { _isDownloading = value; NotifyPropertyChanged(); }
        }

        public ModelConfigTemplate ModelTemplate { get; set; }


        public bool IsTemplate
        {
            get { return _isTemplate; }
            set { _isTemplate = value; NotifyPropertyChanged(); }
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

}
