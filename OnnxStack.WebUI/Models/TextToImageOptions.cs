using OnnxStack.StableDiffusion.Enums;
using System.ComponentModel.DataAnnotations;

namespace OnnxStack.WebUI.Models
{
    public class TextToImageOptions
    {
        [Required]
        [StringLength(512, MinimumLength = 2)]
        public string Prompt { get; set; }

        [StringLength(512)]
        public string NegativePrompt { get; set; }
        public SchedulerType SchedulerType { get; set; }

        [Range(64, 1024)]
        public int Width { get; set; } = 512;

        [Range(64, 1024)]
        public int Height { get; set; } = 512;

        [Range(0, int.MaxValue)]
        public int Seed { get; set; }

        [Range(1, 100)]
        public int InferenceSteps { get; set; } = 30;

        [Range(0f, 40f)]
        public float GuidanceScale { get; set; } = 7.5f;


        public TimestepSpacingType TimestepSpacing { get; set; } = TimestepSpacingType.Linspace;
        public VarianceType VarianceType { get; set; } = VarianceType.FixedSmall;
        public PredictionType PredictionType { get; set; } = PredictionType.Epsilon;
        public AlphaTransformType AlphaTransformType { get; set; } = AlphaTransformType.Cosine;
        public BetaScheduleType BetaSchedule { get; set; } = BetaScheduleType.ScaledLinear;
        public float BetaStart { get; set; } = 0.00085f;
        public float BetaEnd { get; set; } = 0.012f;
        public float MaximumBeta { get; set; } = 0.999f;
        public int TrainTimesteps { get; set; } = 1000;
        public int StepsOffset { get; set; } = 0;
        public bool UseKarrasSigmas { get; set; } = false;
        public bool Thresholding { get; set; } = false;
        public bool ClipSample { get; set; } = false;
        public float ClipSampleRange { get; set; } = 1f;
    }
}
