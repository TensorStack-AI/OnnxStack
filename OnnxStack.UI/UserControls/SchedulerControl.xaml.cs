using OnnxStack.Core;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace OnnxStack.UI.UserControls
{
    /// <summary>
    /// Interaction logic for Parameters.xaml
    /// </summary>
    public partial class SchedulerControl : UserControl, INotifyPropertyChanged
    {

        /// <summary>Initializes a new instance of the <see cref="SchedulerControl" /> class.</summary>
        public SchedulerControl()
        {
            ValidSizes = new ObservableCollection<int>(Constants.ValidSizes);
            RandomSeedCommand = new RelayCommand(RandomSeed);
            ResetParametersCommand = new RelayCommand(ResetParameters);
            InitializeComponent();
        }

        /// <summary>Gets the reset parameters command.</summary>
        /// <value>The reset parameters command.</value>
        public ICommand ResetParametersCommand { get; }
        public ICommand RandomSeedCommand { get; }
        public ObservableCollection<int> ValidSizes { get; }


        public DiffuserType DiffuserType
        {
            get { return (DiffuserType)GetValue(DiffuserTypeProperty); }
            set { SetValue(DiffuserTypeProperty, value); }
        }

        public static readonly DependencyProperty DiffuserTypeProperty =
            DependencyProperty.Register("DiffuserType", typeof(DiffuserType), typeof(SchedulerControl));



        /// <summary>
        /// Gets or sets the SchedulerOptions.
        /// </summary>
        public SchedulerOptionsModel SchedulerOptions
        {
            get { return (SchedulerOptionsModel)GetValue(SchedulerOptionsProperty); }
            set { SetValue(SchedulerOptionsProperty, value); }
        }


        /// <summary>
        /// The SchedulerOptions property
        /// </summary>
        public static readonly DependencyProperty SchedulerOptionsProperty =
            DependencyProperty.Register("SchedulerOptions", typeof(SchedulerOptionsModel), typeof(SchedulerControl));


        /// <summary>
        /// Resets the parameters.
        /// </summary>
        private void ResetParameters()
        {
            SchedulerOptions = new SchedulerOptionsModel();
        }

        private void RandomSeed()
        {
            SchedulerOptions.Seed = Random.Shared.Next();
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
