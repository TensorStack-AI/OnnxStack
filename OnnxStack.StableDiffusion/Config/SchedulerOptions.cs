using OnnxStack.StableDiffusion.Enums;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

namespace OnnxStack.StableDiffusion.Config
{
    public record SchedulerOptions
    {
        /// <summary>
        /// Gets or sets the type of scheduler.
        /// </summary>
        public SchedulerType SchedulerType { get; set; }

        /// <summary>
        /// Gets or sets the height.
        /// </summary>
        /// <value>
        ///  The height of the image. Default is 512 and must be divisible by 64.
        /// </value>
        [Range(0, 4096)]
        public int Height { get; set; } = 512;

        /// <summary>
        /// Gets or sets the width.
        /// </summary>
        /// <value>
        /// The width of the image. Default is 512 and must be divisible by 64.
        /// </value>
        [Range(0, 4096)]
        public int Width { get; set; } = 512;

        /// <summary>
        /// Gets or sets the seed.
        /// </summary>
        /// <value>
        /// If value is set to 0 a random seed is used.
        /// </value>
        [Range(0, int.MaxValue)]
        public int Seed { get; set; }

        /// <summary>
        /// Gets or sets the number inference steps.
        /// </summary>
        /// <value>
        /// The number of steps to run inference for. The more steps the longer it will take to run the inference loop but the image quality should improve.
        /// </value>
        [Range(5, 200)]
        public int InferenceSteps { get; set; } = 30;

        /// <summary>
        /// Gets or sets the guidance scale.
        /// </summary>
        /// <value>
        /// The scale for the classifier-free guidance. The higher the number the more it will try to look like the prompt but the image quality may suffer.
        /// </value>
        [Range(0f, 30f)]
        public float GuidanceScale { get; set; } = 7.5f;

        /// <summary>
        /// Gets or sets the strength use for Image 2 Image
        [Range(0f, 1f)]
        public float Strength { get; set; } = 0.6f;

        [Range(0, int.MaxValue)]
        public int TrainTimesteps { get; set; } = 1000;
        public float BetaStart { get; set; } = 0.00085f;
        public float BetaEnd { get; set; } = 0.012f;
        public IEnumerable<float> TrainedBetas { get; set; }
        public TimestepSpacingType TimestepSpacing { get; set; } = TimestepSpacingType.Linspace;
        public BetaScheduleType BetaSchedule { get; set; } = BetaScheduleType.ScaledLinear;
        public int StepsOffset { get; set; } = 0;
        public bool UseKarrasSigmas { get; set; } = false;
        public VarianceType VarianceType { get; set; } = VarianceType.FixedSmall;
        public float SampleMaxValue { get; set; } = 1.0f;
        public bool Thresholding { get; set; } = false;
        public bool ClipSample { get; set; } = false;
        public float ClipSampleRange { get; set; } = 1f;
        public PredictionType PredictionType { get; set; } = PredictionType.Epsilon;
        public AlphaTransformType AlphaTransformType { get; set; } = AlphaTransformType.Cosine;
        public float MaximumBeta { get; set; } = 0.999f;
        public List<int> Timesteps { get; set; }

        public int OriginalInferenceSteps { get; set; } = 50;

        public float AestheticScore { get; set; } = 6f;
        public float AestheticNegativeScore { get; set; } = 2.5f;

        public float ConditioningScale { get; set; } = 0.7f;

        public int InferenceSteps2 { get; set; } = 10;
        public float GuidanceScale2 { get; set; } = 0;

        public bool IsKarrasScheduler
        {
            get
            {
                return SchedulerType == SchedulerType.LMS
                    || SchedulerType == SchedulerType.KDPM2
                    || SchedulerType == SchedulerType.Euler
                    || SchedulerType == SchedulerType.EulerAncestral;
            }
        }
    }
}