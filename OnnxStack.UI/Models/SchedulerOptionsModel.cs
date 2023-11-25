using OnnxStack.StableDiffusion.Enums;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Runtime.CompilerServices;

namespace OnnxStack.UI.Models
{
    public class SchedulerOptionsModel : INotifyPropertyChanged
    {
        private int _height = 512;
        private int _width = 512;
        private int _seed;
        private int _inferenceSteps = 30;
        private float _guidanceScale = 7.5f;
        private float _strength = 0.75f;
        private float _initialNoiseLevel = 0f;
        private int _trainTimesteps = 1000;
        private float _betaStart = 0.00085f;
        private float _betaEnd = 0.012f;
        private IEnumerable<float> _trainedBetas;
        private TimestepSpacingType _timestepSpacing = TimestepSpacingType.Linspace;
        private BetaScheduleType _betaSchedule = BetaScheduleType.ScaledLinear;
        private int _stepsOffset = 0;
        private bool _useKarrasSigmas = false;
        private VarianceType _varianceType = VarianceType.FixedSmall;
        private float _sampleMaxValue = 1.0f;
        private bool _thresholding = false;
        private bool _clipSample = false;
        private float _clipSampleRange = 1f;
        private PredictionType _predictionType = PredictionType.Epsilon;
        private AlphaTransformType _alphaTransformType = AlphaTransformType.Cosine;
        private float _maximumBeta = 0.999f;
        private int _originalInferenceSteps = 100;
        private SchedulerType _schedulerType;
        private bool _hasChanged;

        /// <summary>
        /// Gets or sets the height.
        /// </summary>
        /// <value>
        ///  The height of the image. Default is 512 and must be divisible by 64.
        /// </value>
        [Range(0, 4096)]
        public int Height
        {
            get { return _height; }
            set { _height = value; NotifyPropertyChanged(); }
        }
        /// <summary>
        /// Gets or sets the width.
        /// </summary>
        /// <value>
        /// The width of the image. Default is 512 and must be divisible by 64.
        /// </value>
        [Range(0, 4096)]
        public int Width
        {
            get
            { return _width; }
            set { _width = value; NotifyPropertyChanged(); }
        }
        /// <summary>
        /// Gets or sets the seed.
        /// </summary>
        /// <value>
        /// If value is set to 0 a random seed is used.
        /// </value>
        [Range(0, int.MaxValue)]
        public int Seed
        {
            get
            { return _seed; }
            set { _seed = value; NotifyPropertyChanged(); }
        }

        /// <summary>
        /// Gets or sets the number inference steps.
        /// </summary>
        /// <value>
        /// The number of steps to run inference for. The more steps the longer it will take to run the inference loop but the image quality should improve.
        /// </value>
        [Range(1, 200)]
        public int InferenceSteps
        {
            get { return _inferenceSteps; }
            set { _inferenceSteps = value; NotifyPropertyChanged(); }
        }
        /// <summary>
        /// Gets or sets the guidance scale.
        /// </summary>
        /// <value>
        /// The scale for the classifier-free guidance. The higher the number the more it will try to look like the prompt but the image quality may suffer.
        /// </value>
        [Range(0f, 30f)]
        public float GuidanceScale
        {
            get { return _guidanceScale; }
            set { _guidanceScale = value; NotifyPropertyChanged(); }
        }
        /// <summary>
        /// Gets or sets the strength use for Image 2 Image
        [Range(0f, 1f)]
        public float Strength
        {
            get { return _strength; }
            set { _strength = value; NotifyPropertyChanged(); }
        }
        /// <summary>
        /// Gets or sets the initial noise level for Image 2 Image
        /// </summary>
        [Range(-1f, 1f)]
        public float InitialNoiseLevel
        {
            get { return _initialNoiseLevel; }
            set { _initialNoiseLevel = value; NotifyPropertyChanged(); }
        }

        [Range(0, int.MaxValue)]
        public int TrainTimesteps
        {
            get { return _trainTimesteps; }
            set { _trainTimesteps = value; NotifyPropertyChanged(); }
        }
        public float BetaStart
        {
            get { return _betaStart; }
            set { _betaStart = value; NotifyPropertyChanged(); }
        }
        public float BetaEnd
        {
            get { return _betaEnd; }
            set { _betaEnd = value; NotifyPropertyChanged(); }
        }
        public IEnumerable<float> TrainedBetas
        {
            get { return _trainedBetas; }
            set { _trainedBetas = value; NotifyPropertyChanged(); }
        }
        public TimestepSpacingType TimestepSpacing
        {
            get { return _timestepSpacing; }
            set { _timestepSpacing = value; NotifyPropertyChanged(); }
        }
        public BetaScheduleType BetaSchedule
        {
            get { return _betaSchedule; }
            set { _betaSchedule = value; NotifyPropertyChanged(); }
        }
        public int StepsOffset
        {
            get { return _stepsOffset; }
            set { _stepsOffset = value; NotifyPropertyChanged(); }
        }
        public bool UseKarrasSigmas
        {
            get { return _useKarrasSigmas; }
            set { _useKarrasSigmas = value; NotifyPropertyChanged(); }
        }
        public VarianceType VarianceType
        {
            get { return _varianceType; }
            set { _varianceType = value; NotifyPropertyChanged(); }
        }
        public float SampleMaxValue
        {
            get { return _sampleMaxValue; }
            set { _sampleMaxValue = value; NotifyPropertyChanged(); }
        }
        public bool Thresholding
        {
            get { return _thresholding; }
            set { _thresholding = value; NotifyPropertyChanged(); }
        }
        public bool ClipSample
        {
            get { return _clipSample; }
            set { _clipSample = value; NotifyPropertyChanged(); }
        }
        public float ClipSampleRange
        {
            get { return _clipSampleRange; }
            set { _clipSampleRange = value; NotifyPropertyChanged(); }
        }
        public PredictionType PredictionType
        {
            get { return _predictionType; }
            set { _predictionType = value; NotifyPropertyChanged(); }
        }
        public AlphaTransformType AlphaTransformType
        {
            get { return _alphaTransformType; }
            set { _alphaTransformType = value; NotifyPropertyChanged(); }
        }

        public float MaximumBeta
        {
            get { return _maximumBeta; }
            set { _maximumBeta = value; NotifyPropertyChanged(); }
        }

        public int OriginalInferenceSteps
        {
            get { return _originalInferenceSteps; }
            set { _originalInferenceSteps = value; NotifyPropertyChanged(); }
        }

        public SchedulerType SchedulerType
        {
            get { return _schedulerType; }
            set { _schedulerType = value; NotifyPropertyChanged(); }
        }

        public bool HasChanged
        {
            get { return _hasChanged; }
            set { _hasChanged = value; NotifyPropertyChanged(); }
        }



        #region INotifyPropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;
        public void NotifyPropertyChanged([CallerMemberName] string property = "")
        {
            if (!property.Equals(nameof(HasChanged)) && !HasChanged)
                HasChanged = true;

            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(property));
        }
        #endregion

    }
}