using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.StableDiffusion.Config;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Common
{
    public interface ISchedulerService
    {

        /// <summary>
        /// Runs the specified Scheduler with the prompt inputs provided.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <returns></returns>
        Task<DenseTensor<float>> RunAsync(PromptOptions prompt, SchedulerOptions options, Action<int, int> progress = null, CancellationToken cancellationToken = default);
    }
}