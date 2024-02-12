using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.FeatureExtractor.Common;
using OnnxStack.StableDiffusion.Enums;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace OnnxStack.UI.Models
{
    public class UpdateFeatureExtractorModelSetViewModel : INotifyPropertyChanged
    {
        private string _name;
        private int _deviceId;
        private int _interOpNumThreads;
        private int _intraOpNumThreads;
        private ExecutionMode _executionMode;
        private ExecutionProvider _executionProvider;
        private string _modelFile;
        private ControlNetType? _controlNetType;
        private int _sampleSize;
        private bool _normalize;
        private int _channels;

        public string Name
        {
            get { return _name; }
            set { _name = value; NotifyPropertyChanged(); }
        }

        public ControlNetType? ControlNetType
        {
            get { return _controlNetType; }
            set { _controlNetType = value; NotifyPropertyChanged(); }
        }

        public int SampleSize
        {
            get { return _sampleSize; }
            set { _sampleSize = value; NotifyPropertyChanged(); }
        }

        public bool Normalize
        {
            get { return _normalize; }
            set { _normalize = value; NotifyPropertyChanged(); }
        }

        public int Channels
        {
            get { return _channels; }
            set { _channels = value; NotifyPropertyChanged(); }
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

        public string ModelFile
        {
            get { return _modelFile; }
            set { _modelFile = value; NotifyPropertyChanged(); }
        }


        public static UpdateFeatureExtractorModelSetViewModel FromModelSet(FeatureExtractorModelSet modelset, ControlNetType? controlNetType)
        {
            return new UpdateFeatureExtractorModelSetViewModel
            {
                Name = modelset.Name,
                DeviceId = modelset.DeviceId,
                ExecutionMode = modelset.ExecutionMode,
                ExecutionProvider = modelset.ExecutionProvider,
                InterOpNumThreads = modelset.InterOpNumThreads,
                IntraOpNumThreads = modelset.IntraOpNumThreads,

                ControlNetType = controlNetType,
                ModelFile = modelset.FeatureExtractorConfig.OnnxModelPath,
                Normalize = modelset.FeatureExtractorConfig.Normalize,
                SampleSize = modelset.FeatureExtractorConfig.SampleSize,
                Channels = modelset.FeatureExtractorConfig.Channels,
            };
        }

        public static FeatureExtractorModelSet ToModelSet(UpdateFeatureExtractorModelSetViewModel modelset)
        {
            return new FeatureExtractorModelSet
            {
                IsEnabled = true,
                Name = modelset.Name,
                DeviceId = modelset.DeviceId,
                ExecutionMode = modelset.ExecutionMode,
                ExecutionProvider = modelset.ExecutionProvider,
                InterOpNumThreads = modelset.InterOpNumThreads,
                IntraOpNumThreads = modelset.IntraOpNumThreads,
                FeatureExtractorConfig = new FeatureExtractorModelConfig
                {
                    Channels = modelset.Channels,
                    Normalize = modelset.Normalize,
                    SampleSize = modelset.SampleSize,
                    OnnxModelPath = modelset.ModelFile
                }
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
