using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace OnnxStack.UI.Models
{
    public record UpscaleInfoModel : INotifyPropertyChanged
    {
        private int _inputWidth;
        private int _inputHeight;
        private int _outputWidth;
        private int _outputHeight;
        private int _sampleSize;
        private int _scaleFactor;

        public int InputWidth
        {
            get { return _inputWidth; }
            set { _inputWidth = value; NotifyPropertyChanged(); Update(); }
        }

        public int InputHeight
        {
            get { return _inputHeight; }
            set { _inputHeight = value; NotifyPropertyChanged(); Update(); }
        }

        public int OutputWidth
        {
            get { return _outputWidth; }
            set { _outputWidth = value; NotifyPropertyChanged(); }
        }

        public int OutputHeight
        {
            get { return _outputHeight; }
            set { _outputHeight = value; NotifyPropertyChanged(); }
        }

        public int SampleSize
        {
            get { return _sampleSize; }
            set { _sampleSize = value; NotifyPropertyChanged(); }
        }


        public int ScaleFactor
        {
            get { return _scaleFactor; }
            set { _scaleFactor = value; NotifyPropertyChanged(); Update(); }
        }

        public void Update()
        {
            OutputWidth = _inputWidth * _scaleFactor;
            OutputHeight = _inputHeight * _scaleFactor;
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
