using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace OnnxStack.UI.Models
{
    public class SchedulerOptionsConfig : INotifyPropertyChanged
    {

        private int _stepsMin = 4;
        private int _stepsMax = 100;

        public int StepsMin
        {
            get { return _stepsMin; }
            set { _stepsMin = value; NotifyPropertyChanged(); }
        }

        public int StepsMax
        {
            get { return _stepsMax; }
            set { _stepsMax = value; NotifyPropertyChanged(); }
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
