using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Config;
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
        private ObservableCollection<ModelFileViewModel> _modelFiles;
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

        public ObservableCollection<ModelFileViewModel> ModelFiles
        {
            get { return _modelFiles; }
            set { _modelFiles = value; NotifyPropertyChanged(); }
        }


        public static UpdateUpscaleModelSetViewModel FromModelSet(UpscaleModelSet modelset)
        {
            return new UpdateUpscaleModelSetViewModel
            {
                Name = modelset.Name,
                SampleSize = modelset.SampleSize,
                ScaleFactor = modelset.ScaleFactor,
                Channels = modelset.Channels,

                DeviceId = modelset.DeviceId,
                ExecutionMode = modelset.ExecutionMode,
                ExecutionProvider = modelset.ExecutionProvider,
                InterOpNumThreads = modelset.InterOpNumThreads,
                IntraOpNumThreads = modelset.IntraOpNumThreads,
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

        public static UpscaleModelSet ToModelSet(UpdateUpscaleModelSetViewModel modelset)
        {
            return new UpscaleModelSet
            {
                IsEnabled = true,
                Name = modelset.Name,
                SampleSize = modelset.SampleSize,
                ScaleFactor = modelset.ScaleFactor,
                Channels = modelset.Channels,
                DeviceId = modelset.DeviceId,
                ExecutionMode = modelset.ExecutionMode,
                ExecutionProvider = modelset.ExecutionProvider,
                InterOpNumThreads = modelset.InterOpNumThreads,
                IntraOpNumThreads = modelset.IntraOpNumThreads,
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
