using Models;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Enums;
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
        private SchedulerOptionsConfig _optionsConfig = new();

        /// <summary>Initializes a new instance of the <see cref="SchedulerControl" /> class.</summary>
        public SchedulerControl()
        {
            ValidSizes = new ObservableCollection<int>(Constants.ValidSizes);
            RandomSeedCommand = new RelayCommand(RandomSeed);
            ResetParametersCommand = new RelayCommand(ResetParameters);
            InitializeComponent();
        }

        public ICommand ResetParametersCommand { get; }
        public ICommand RandomSeedCommand { get; }
        public ObservableCollection<int> ValidSizes { get; }

        /// <summary>
        /// Gets or sets the selected model.
        /// </summary>
        public ModelOptionsModel SelectedModel
        {
            get { return (ModelOptionsModel)GetValue(SelectedModelProperty); }
            set { SetValue(SelectedModelProperty, value); }
        }
        public static readonly DependencyProperty SelectedModelProperty =
            DependencyProperty.Register("SelectedModel", typeof(ModelOptionsModel), typeof(SchedulerControl), new PropertyMetadata((d, e) =>
            {
                if (d is SchedulerControl schedulerControl)
                    schedulerControl.OnModelChanged(e.NewValue as ModelOptionsModel);
            }));


        /// <summary>
        /// Gets or sets the type of the diffuser.
        /// </summary>
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
        public static readonly DependencyProperty SchedulerOptionsProperty =
            DependencyProperty.Register("SchedulerOptions", typeof(SchedulerOptionsModel), typeof(SchedulerControl));


        /// <summary>
        /// Gets or sets the options configuration.
        /// </summary>
        public SchedulerOptionsConfig OptionsConfig
        {
            get { return _optionsConfig; }
            set { _optionsConfig = value; NotifyPropertyChanged(); }
        }


        /// <summary>
        /// Called when the selected model has changed.
        /// </summary>
        /// <param name="modelOptionsModel">The model options model.</param>
        private void OnModelChanged(ModelOptionsModel model)
        {
            if (model is null)
                return;

            if (model.ModelOptions.PipelineType == DiffuserPipelineType.StableDiffusion)
            {
                OptionsConfig.StepsMin = 4;
                OptionsConfig.StepsMax = 100;
                SchedulerOptions.InferenceSteps = 30;
            }
            else if (model.ModelOptions.PipelineType == DiffuserPipelineType.LatentConsistency)
            {
                OptionsConfig.StepsMin = 1;
                OptionsConfig.StepsMax = 50;
                SchedulerOptions.InferenceSteps = 6;
            }
        }


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
