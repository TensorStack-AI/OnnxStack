using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Results;
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

        public Task<ImageResult> TextToImageFile(PromptOptions prompt, string outputFile)
        {
            return TextToImageFileInternal(prompt, new SchedulerOptions(), outputFile);
        }

        public Task<ImageResult> TextToImageFile(PromptOptions prompt, SchedulerOptions options, string outputFile)
        {
            return TextToImageFileInternal(prompt, options, outputFile);
        }


        private async Task<ImageResult> TextToImageInternal(PromptOptions prompt, SchedulerOptions options)
        {
            var imageTensorData = await _schedulerService.RunAsync(prompt, options).ConfigureAwait(false);
            return ImageHelpers.TensorToImage(options, imageTensorData);
        }

        private async Task<ImageResult> TextToImageFileInternal(PromptOptions prompt, SchedulerOptions options, string outputFile)
        {
            var result = await TextToImageInternal(prompt, options);
            if (result is null)
                return null;

            await result.SaveAsync(outputFile);
            return result;
        }
    }
}
