using NumSharp;
using System.Collections.Generic;

namespace OnnxStack.StableDiffusion.Config
{
    public class SchedulerOptions
    {
        public int TrainTimesteps { get; set; } = 1000;
        public float BetaStart { get; set; } = 0.00085f;
        public float BetaEnd { get; set; } = 0.012f;
        public IEnumerable<float> TrainedBetas { get; set; }
        public TimestepSpacing TimestepSpacing { get; set; } = TimestepSpacing.Linspace;
        public BetaSchedule BetaSchedule { get; set; } = BetaSchedule.ScaledLinear;
        public int StepsOffset { get; set; } = 0;
        public bool UseKarrasSigmas { get; set; } = false;
        public VarianceType VarianceType { get; internal set; } = VarianceType.FixedSmall;
        public float SampleMaxValue { get; set; } = 1.0f;
        public bool Thresholding { get; internal set; } = false;
        public bool ClipSample { get; internal set; } = false;
        public float ClipSampleRange { get; internal set; } = 1f;
        public PredictionType PredictionType { get; internal set; } = PredictionType.Epsilon;
        public AlphaTransformType AlphaTransformType { get; set; } = AlphaTransformType.Cosine;
        public float MaximumBeta { get; set; } = 0.999f;

    }
}