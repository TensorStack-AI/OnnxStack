using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
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
        public static List<SchedulerOptions> GenerateBatch(IModelOptions modelOptions, BatchOptions batchOptions, SchedulerOptions schedulerOptions)
        {
            if (batchOptions.BatchType == BatchOptionType.Seed)
            {
                return Enumerable.Range(0, Math.Max(1, batchOptions.Count))
                    .Select(x => Random.Shared.Next())
                    .Select(x => schedulerOptions with { Seed = x })
                    .ToList();
            }
            else if (batchOptions.BatchType == BatchOptionType.Step)
            {
                return Enumerable.Range(Math.Max(0, (int)batchOptions.ValueFrom), Math.Max(1, (int)batchOptions.ValueTo))
                  .Select(x => schedulerOptions with { InferenceSteps = x })
                  .ToList();
            }
            else if (batchOptions.BatchType == BatchOptionType.Guidance)
            {
                var totalIncrements = (int)Math.Max(1, (batchOptions.ValueTo - batchOptions.ValueFrom) / batchOptions.Increment);
                return Enumerable.Range(0, totalIncrements)
                  .Select(x => schedulerOptions with { GuidanceScale = batchOptions.ValueFrom + (batchOptions.Increment * x) })
                  .ToList();
            }
            else if (batchOptions.BatchType == BatchOptionType.Strength)
            {
                var totalIncrements = (int)Math.Max(1, (batchOptions.ValueTo - batchOptions.ValueFrom) / batchOptions.Increment);
                return Enumerable.Range(0, totalIncrements)
                  .Select(x => schedulerOptions with { Strength = batchOptions.ValueFrom + (batchOptions.Increment * x) })
                  .ToList();
            }
            else if (batchOptions.BatchType == BatchOptionType.Scheduler)
            {
                return modelOptions.PipelineType.GetSchedulerTypes()
                  .Select(x => schedulerOptions with { SchedulerType = x })
                  .ToList();
            }
            return new List<SchedulerOptions>();
        }
    }
}
