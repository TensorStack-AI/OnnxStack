using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IPipeline
    {
        IReadOnlyList<DiffuserType> SupportedDiffusers { get; }
        IReadOnlyList<SchedulerType> SupportedSchedulers { get; }
        Task LoadAsync();
        Task UnloadAsync();
        void ValidateInputs(PromptOptions promptOptions, SchedulerOptions schedulerOptions);
        Task<DenseTensor<float>> RunAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);
        IAsyncEnumerable<BatchResult> RunBatchAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);
    }
}