using OnnxStack.Core;
using OnnxStack.StableDiffusion;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
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
        private ObservableCollection<SchedulerType> _schedulerTypes = new();

        /// <summary>Initializes a new instance of the <see cref="SchedulerControl" /> class.</summary>
        public SchedulerControl()
        {
            ValidSizes = new ObservableCollection<int>(new [] { 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024 });
            NewSeedCommand = new RelayCommand(NewSeed);
            RandomSeedCommand = new RelayCommand(RandomSeed);
            ResetParametersCommand = new RelayCommand(ResetParameters);
            InitializeComponent();
        }

        public ICommand ResetParametersCommand { get; }
        public ICommand NewSeedCommand { get; }
        public ICommand RandomSeedCommand { get; }
        public ObservableCollection<int> ValidSizes { get; }

        public ObservableCollection<SchedulerType> SchedulerTypes
        {
            get { return _schedulerTypes; }
            set { _schedulerTypes = value; NotifyPropertyChanged(); }
        }

        /// <summary>
        /// Gets or sets the selected model.
        /// </summary>
        public StableDiffusionModelSetViewModel SelectedModel
        {
            get { return (StableDiffusionModelSetViewModel)GetValue(SelectedModelProperty); }
            set { SetValue(SelectedModelProperty, value); }
        }
        public static readonly DependencyProperty SelectedModelProperty =
            DependencyProperty.Register("SelectedModel", typeof(StableDiffusionModelSetViewModel), typeof(SchedulerControl), new PropertyMetadata((d, e) =>
            {
                if (d is SchedulerControl schedulerControl)
                    schedulerControl.OnModelChanged(e.NewValue as StableDiffusionModelSetViewModel);
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

    
        public bool IsGenerating
        {
            get { return (bool)GetValue(IsGeneratingProperty); }
            set { SetValue(IsGeneratingProperty, value); }
        }
        public static readonly DependencyProperty IsGeneratingProperty =
            DependencyProperty.Register("IsGenerating", typeof(bool), typeof(SchedulerControl));



        /// <summary>
        /// Called when the selected model has changed.
        /// </summary>
        /// <param name="modelOptionsModel">The model options model.</param>
        private void OnModelChanged(StableDiffusionModelSetViewModel model)
        {
            if (model is null)
                return;

            SchedulerTypes.Clear();
            foreach (SchedulerType type in GetSchedulerTypes(model.ModelSet.PipelineType))
                SchedulerTypes.Add(type);

            SchedulerOptions.Width = 512;
            SchedulerOptions.Height = 512;
            if (model.ModelSet.PipelineType == DiffuserPipelineType.StableDiffusion)
            {
                SchedulerOptions.OriginalInferenceSteps = 100;
                SchedulerOptions.InferenceSteps = 30;
                SchedulerOptions.GuidanceScale = 7.5f;
                SchedulerOptions.SchedulerType = SchedulerType.DDIM;
            }
            else if (model.ModelSet.PipelineType == DiffuserPipelineType.LatentConsistency)
            {
                SchedulerOptions.OriginalInferenceSteps = 50;
                SchedulerOptions.InferenceSteps = 6;
                SchedulerOptions.GuidanceScale = 1f;
                SchedulerOptions.SchedulerType = SchedulerType.LCM;
            }
            else if (model.ModelSet.PipelineType == DiffuserPipelineType.LatentConsistencyXL)
            {
                SchedulerOptions.OriginalInferenceSteps = 50;
                SchedulerOptions.InferenceSteps = 6;
                SchedulerOptions.GuidanceScale = 1f;
                SchedulerOptions.Width = 1024;
                SchedulerOptions.Height = 1024;
                SchedulerOptions.SchedulerType = SchedulerType.LCM;
            }
            else if (model.ModelSet.PipelineType == DiffuserPipelineType.InstaFlow)
            {
                SchedulerOptions.InferenceSteps = 1;
                SchedulerOptions.GuidanceScale = 0f;
                SchedulerOptions.SchedulerType = SchedulerType.InstaFlow;
            }
            else if (model.ModelSet.PipelineType == DiffuserPipelineType.StableDiffusionXL)
            {
                SchedulerOptions.OriginalInferenceSteps = 100;
                SchedulerOptions.InferenceSteps = 30;
                SchedulerOptions.GuidanceScale = 5f;
                SchedulerOptions.Width = 1024;
                SchedulerOptions.Height = 1024;
                SchedulerOptions.SchedulerType = SchedulerType.EulerAncestral;
            }
        }


        /// <summary>
        /// Resets the parameters.
        /// </summary>
        private void ResetParameters()
        {
            SchedulerOptions = new SchedulerOptionsModel
            {
               SchedulerType = GetSchedulerTypes(SelectedModel.ModelSet.PipelineType).First()
            };
        }

        private void NewSeed()
        {
            SchedulerOptions.Seed = Random.Shared.Next();
        }

        private void RandomSeed()
        {
            SchedulerOptions.Seed = 0;
        }


        /// <summary>
        /// TODO: Get from pipeline
        /// </summary>
        /// <param name="pipelineType">Type of the pipeline.</param>
        /// <returns></returns>
        private static SchedulerType[] GetSchedulerTypes(DiffuserPipelineType pipelineType)
        {
            switch (pipelineType)
            {
                case DiffuserPipelineType.StableDiffusion:
                case DiffuserPipelineType.StableDiffusionXL:
                    return new[]
                    {
                        SchedulerType.LMS,
                        SchedulerType.Euler,
                        SchedulerType.EulerAncestral,
                        SchedulerType.DDPM,
                        SchedulerType.DDIM,
                        SchedulerType.KDPM2
                    };
                case DiffuserPipelineType.LatentConsistency:
                case DiffuserPipelineType.LatentConsistencyXL:
                    return new[]
                    {
                        SchedulerType.LCM
                    };
                case DiffuserPipelineType.InstaFlow:
                    return new[]
                    {
                        SchedulerType.InstaFlow
                    };
                default:
                    return default;
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
