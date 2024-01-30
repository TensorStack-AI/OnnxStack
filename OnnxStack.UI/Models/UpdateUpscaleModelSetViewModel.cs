using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.ImageUpscaler.Common;
using OnnxStack.UI.Views;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;

namespace OnnxStack.UI.Models
{
    public class UpdateUpscaleModelSetViewModel : INotifyPropertyChanged
    {
        private string _name;
        private int _deviceId;
        private int _interOpNumThreads;
        private int _intraOpNumThreads;
        private ExecutionMode _executionMode;
        private ExecutionProvider _executionProvider;
        private string _modelFile;
        private int _channels;
        private int _scaleFactor;
        private int _sampleSize;

        public string Name
        {
            get { return _name; }
            set { _name = value; NotifyPropertyChanged(); }
        }

        public int Channels
        {
            get { return _channels; }
            set { _channels = value; NotifyPropertyChanged(); }
        }
        public int ScaleFactor
        {
            get { return _scaleFactor; }
            set { _scaleFactor = value; NotifyPropertyChanged(); }
        }

        public int SampleSize
        {
            get { return _sampleSize; }
            set { _sampleSize = value; NotifyPropertyChanged(); }
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


        public static UpdateUpscaleModelSetViewModel FromModelSet(UpscaleModelSet modelset)
        {
            return new UpdateUpscaleModelSetViewModel
            {
                Name = modelset.Name,
                DeviceId = modelset.DeviceId,
                ExecutionMode = modelset.ExecutionMode,
                ExecutionProvider = modelset.ExecutionProvider,
                InterOpNumThreads = modelset.InterOpNumThreads,
                IntraOpNumThreads = modelset.IntraOpNumThreads,
                SampleSize = modelset.UpscaleModelConfig.SampleSize,
                ScaleFactor = modelset.UpscaleModelConfig.ScaleFactor,
                Channels = modelset.UpscaleModelConfig.Channels,
                ModelFile = modelset.UpscaleModelConfig.OnnxModelPath
            };
        }

        public static UpscaleModelSet ToModelSet(UpdateUpscaleModelSetViewModel modelset)
        {
            return new UpscaleModelSet
            {
                IsEnabled = true,
                Name = modelset.Name,
                DeviceId = modelset.DeviceId,
                ExecutionMode = modelset.ExecutionMode,
                ExecutionProvider = modelset.ExecutionProvider,
                InterOpNumThreads = modelset.InterOpNumThreads,
                IntraOpNumThreads = modelset.IntraOpNumThreads,
                UpscaleModelConfig = new UpscaleModelConfig
                {
                    OnnxModelPath = modelset.ModelFile,
                    Channels = modelset.Channels,
                    ScaleFactor = modelset.ScaleFactor,
                    SampleSize = modelset.SampleSize
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
