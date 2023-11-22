using OnnxStack.StableDiffusion.Enums;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace Models
{
    public class BatchOptionsModel : INotifyPropertyChanged
    {
        private float _valueTo;
        private float _valueFrom;
        private float _increment = 1;
        private BatchOptionType _batchType;
        private bool _isAutomationEnabled;
        private int _stepValue;
        private int _stepsValue = 1;
        private int _batchValue;
        private int _batchsValue = 1;
        private bool _disableHistory = true;
        private bool _isRealtimeEnabled;

        public BatchOptionType BatchType
        {
            get { return _batchType; }
            set { _batchType = value; NotifyPropertyChanged(); }
        }

        public float ValueTo
        {
            get { return _valueTo; }
            set { _valueTo = value; NotifyPropertyChanged(); }
        }

        public float ValueFrom
        {
            get { return _valueFrom; }
            set { _valueFrom = value; NotifyPropertyChanged(); }
        }

        public float Increment
        {
            get { return _increment; }
            set { _increment = value; NotifyPropertyChanged(); }
        }

        public int StepValue
        {
            get { return _stepValue; }
            set { _stepValue = value; NotifyPropertyChanged(); }
        }

        public int StepsValue
        {
            get { return _stepsValue; }
            set { _stepsValue = value; NotifyPropertyChanged(); }
        }

        public int BatchValue
        {
            get { return _batchValue; }
            set { _batchValue = value; NotifyPropertyChanged(); }
        }

        public int BatchsValue
        {
            get { return _batchsValue; }
            set { _batchsValue = value; NotifyPropertyChanged(); }
        }

        public bool DisableHistory
        {
            get { return _disableHistory; }
            set { _disableHistory = value; NotifyPropertyChanged(); }
        }

        public bool IsAutomationEnabled
        {
            get { return _isAutomationEnabled; }
            set
            {
                _isAutomationEnabled = value;
                if (_isAutomationEnabled)
                    IsRealtimeEnabled = false;
                NotifyPropertyChanged();
            }
        }

        public bool IsRealtimeEnabled
        {
            get { return _isRealtimeEnabled; }
            set
            {
                _isRealtimeEnabled = value;
                if (_isRealtimeEnabled)
                    IsAutomationEnabled = false;
                NotifyPropertyChanged();
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
}
