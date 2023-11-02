using Models;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
using System.Collections.ObjectModel;
using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Linq;

namespace OnnxStack.UI.UserControls
{
    /// <summary>
    /// Interaction logic for Parameters.xaml
    /// </summary>
    public partial class PromptControl : UserControl, INotifyPropertyChanged
    {
        private ObservableCollection<SchedulerType> _schedulerTypes = new();

        /// <summary>Initializes a new instance of the <see cref="PromptControl" /> class.</summary>
        public PromptControl()
        {
            ResetParametersCommand = new RelayCommand(ResetParameters);
            InitializeComponent();
        }

        /// <summary>Gets the reset parameters command.</summary>
        /// <value>The reset parameters command.</value>
        public ICommand ResetParametersCommand { get; }


        /// <summary>
        /// Gets or sets the PromptOptions.
        /// </summary>
        public PromptOptionsModel PromptOptions
        {
            get { return (PromptOptionsModel)GetValue(PromptOptionsProperty); }
            set { SetValue(PromptOptionsProperty, value); }
        }


        /// <summary>
        /// The PromptOptions property
        /// </summary>
        public static readonly DependencyProperty PromptOptionsProperty =
            DependencyProperty.Register("PromptOptions", typeof(PromptOptionsModel), typeof(PromptControl));


        public ModelOptionsModel SelectedModel
        {
            get { return (ModelOptionsModel)GetValue(SelectedModelProperty); }
            set { SetValue(SelectedModelProperty, value); }
        }

        public static readonly DependencyProperty SelectedModelProperty =
            DependencyProperty.Register("SelectedModel", typeof(ModelOptionsModel), typeof(PromptControl), new PropertyMetadata((d, e) =>
            {
                if (d is PromptControl schedulerControl)
                    schedulerControl.OnModelChanged(e.NewValue as ModelOptionsModel);
            }));

        public ObservableCollection<SchedulerType> SchedulerTypes
        {
            get { return _schedulerTypes; }
            set { _schedulerTypes = value; NotifyPropertyChanged(); }
        }


        /// <summary>
        /// Called when the selected model has changed.
        /// </summary>
        /// <param name="modelOptionsModel">The model options model.</param>
        private void OnModelChanged(ModelOptionsModel model)
        {
            SchedulerTypes.Clear();
            if (model.ModelOptions.PipelineType == DiffuserPipelineType.StableDiffusion)
            {
                foreach (SchedulerType type in Enum.GetValues<SchedulerType>().Where(x => x != SchedulerType.LCM))
                    SchedulerTypes.Add(type);
            }
            else if (model.ModelOptions.PipelineType == DiffuserPipelineType.LatentConsistency)
            {
                SchedulerTypes.Add(SchedulerType.LCM);
            }

            PromptOptions.SchedulerType = SchedulerTypes.FirstOrDefault();
        }

        /// <summary>
        /// Resets the parameters.
        /// </summary>
        private void ResetParameters()
        {
            PromptOptions = new PromptOptionsModel();
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
