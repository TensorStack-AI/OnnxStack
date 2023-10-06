using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Results;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Services
{
    public sealed class StableDiffusionService : IStableDiffusionService
    {
        private readonly ISchedulerService _schedulerService;

        public StableDiffusionService(ISchedulerService schedulerService)
        {
            _schedulerService = schedulerService;
        }

        public Task<ImageResult> TextToImage(PromptOptions prompt)
        {
            return TextToImageInternal(prompt, new SchedulerOptions());
        }

        public Task<ImageResult> TextToImage(PromptOptions prompt, SchedulerOptions options)
        {
            return TextToImageInternal(prompt, options);
        }

        public Task<ImageResult> TextToImage(PromptOptions prompt, SchedulerOptions options, Action<int, int> progress = null, CancellationToken cancellationToken = default)
        {
            return TextToImageInternal(prompt, options, progress, cancellationToken);
        }




        public Task<ImageResult> TextToImageFile(PromptOptions prompt, string outputFile)
        {
            return TextToImageFileInternal(prompt, new SchedulerOptions(), outputFile);
        }

        public Task<ImageResult> TextToImageFile(PromptOptions prompt, SchedulerOptions options, string outputFile)
        {
            return TextToImageFileInternal(prompt, options, outputFile);
        }

        public Task<ImageResult> TextToImageFile(PromptOptions prompt, SchedulerOptions options, string outputFile, Action<int, int> progress = null, CancellationToken cancellationToken = default)
        {
            return TextToImageFileInternal(prompt, options, outputFile, progress, cancellationToken);
        }


        private async Task<ImageResult> TextToImageInternal(PromptOptions prompt, SchedulerOptions options, Action<int, int> progress = null, CancellationToken cancellationToken = default)
        {
            var imageTensorData = await _schedulerService.RunAsync(prompt, options, progress, cancellationToken).ConfigureAwait(false);
            return ImageHelpers.TensorToImage(options, imageTensorData);
        }

        private async Task<ImageResult> TextToImageFileInternal(PromptOptions prompt, SchedulerOptions options, string outputFile, Action<int, int> progress = null, CancellationToken cancellationToken = default)
        {
            var result = await TextToImageInternal(prompt, options, progress, cancellationToken);
            if (result is null)
                return null;

            await result.SaveAsync(outputFile);
            return result;
        }
    }
}
