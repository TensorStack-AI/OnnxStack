using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Schedulers.InstaFlow
{
    internal class InstaFlowScheduler : SchedulerBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="InstaFlowScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        public InstaFlowScheduler() : this(new SchedulerOptions()) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="InstaFlowScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        public InstaFlowScheduler(SchedulerOptions options) : base(options) { }

        protected override void Initialize()
        {
            SetInitNoiseSigma(1f);
        }

        protected override int[] SetTimesteps()
        {
            var timesteps = new List<double>();
            for (int i = 0; i < Options.InferenceSteps; i++)
            {
                double timestep = (1.0 - (double)i / Options.InferenceSteps) * 1000.0;
                timesteps.Add(timestep);
            }

            return timesteps
                .Select(x => (int)x)
                .OrderByDescending(x => x)
                .ToArray();
        }



        public override DenseTensor<float> ScaleInput(DenseTensor<float> sample, int timestep)
        {
            return sample;
        }


        public override SchedulerStepResult Step(DenseTensor<float> modelOutput, int timestep, DenseTensor<float> sample, int order = 4)
        {
            return new SchedulerStepResult(sample);
        }


        public override DenseTensor<float> AddNoise(DenseTensor<float> originalSamples, DenseTensor<float> noise, IReadOnlyList<int> timesteps)
        {
            return originalSamples;
        }
    }
}
