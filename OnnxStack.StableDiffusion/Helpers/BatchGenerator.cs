using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Pipelines;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxStack.StableDiffusion.Helpers
{
    public static class BatchGenerator
    {
        /// <summary>
        /// Generates the batch of SchedulerOptions fo batch processing.
        /// </summary>
        /// <param name="batchOptions">The batch options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <returns></returns>
        public static List<SchedulerOptions> GenerateBatch(IPipeline pipeline, GenerateBatchOptions batchOptions, SchedulerOptions schedulerOptions)
        {
            if (batchOptions.BatchType == BatchOptionType.Seed)
            {
                var seed = schedulerOptions.Seed == 0 ? Random.Shared.Next() : schedulerOptions.Seed;
                if (batchOptions.ValueTo <= 1)
                    return [schedulerOptions with { Seed = seed }];

                var random = new Random(seed);
                return Enumerable.Range(0, Math.Max(1, (int)batchOptions.ValueTo - 1))
                    .Select(x => random.Next())
                    .Prepend(seed)
                    .Select(x => schedulerOptions with { Seed = x })
                    .ToList();
            }

            if (batchOptions.BatchType == BatchOptionType.Scheduler)
            {
                return pipeline.SupportedSchedulers
                  .Select(x => schedulerOptions with { SchedulerType = x })
                  .ToList();
            }

            var totalIncrements = (int)Math.Max(1, (batchOptions.ValueTo - batchOptions.ValueFrom) / batchOptions.Increment) + 1;
            if (batchOptions.BatchType == BatchOptionType.Step)
            {
                return Enumerable.Range(0, totalIncrements)
                   .Select(x => schedulerOptions with { InferenceSteps = (int)(batchOptions.ValueFrom + (batchOptions.Increment * x)) })
                   .ToList();
            }

            if (batchOptions.BatchType == BatchOptionType.Guidance)
            {
                return Enumerable.Range(0, totalIncrements)
                  .Select(x => schedulerOptions with { GuidanceScale = batchOptions.ValueFrom + (batchOptions.Increment * x) })
                  .ToList();
            }

            if (batchOptions.BatchType == BatchOptionType.Strength)
            {
                return Enumerable.Range(0, totalIncrements)
                  .Select(x => schedulerOptions with { Strength = batchOptions.ValueFrom + (batchOptions.Increment * x) })
                  .ToList();
            }

            if (batchOptions.BatchType == BatchOptionType.ConditioningScale)
            {
                return Enumerable.Range(0, totalIncrements)
                  .Select(x => schedulerOptions with { ConditioningScale = batchOptions.ValueFrom + (batchOptions.Increment * x) })
                  .ToList();
            }
            return new List<SchedulerOptions>();
        }
    }
}
