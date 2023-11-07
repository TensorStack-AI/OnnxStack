using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace OnnxStack.UI.Views
{
    public class ModelFileViewModel : INotifyPropertyChanged
    {
        private OnnxModelType _type;
        private string _onnnxModelPath;
        private int? _deviceId;
        private int? _interOpNumThreads;
        private int? _intraOpNumThreads;
        private ExecutionMode? _executionMode;
        private ExecutionProvider? _executionProvider;
        private bool _isOverrideEnabled;

        public string OnnxModelPath
        {
            get { return _onnnxModelPath; }
            set { _onnnxModelPath = value; NotifyPropertyChanged(); }
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
