using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Services
{
    public class StableDiffusionService : IStableDiffusionService
    {
        private readonly IImageService _imageService;
        private readonly IInferenceService _inferenceService;
        private readonly OnnxStackConfig _configuration;

        public StableDiffusionService(OnnxStackConfig configuration)
        {
            _configuration = configuration;
            _imageService = new ImageService();
            _inferenceService = new InferenceService(_configuration);
        }

        public Task<Image<Rgba32>> TextToImage(string prompt)
        {
            return TextToImageInternal(prompt, null, new SchedulerConfig());
        }

        public Task<Image<Rgba32>> TextToImage(string prompt, string negativePrompt)
        {
            return TextToImageInternal(prompt, negativePrompt, new SchedulerConfig());
        }

        public Task<Image<Rgba32>> TextToImage(string prompt, SchedulerConfig schedulerConfig)
        {
            return TextToImageInternal(prompt, null, schedulerConfig);
        }

        public Task<Image<Rgba32>> TextToImage(string prompt, string negativePrompt, SchedulerConfig schedulerConfig)
        {
            return TextToImageInternal(prompt, negativePrompt, schedulerConfig);
        }


        public Task<bool> TextToImageFile(string prompt, string filename)
        {
            return TextToImageFileInternal(prompt, null, filename, new SchedulerConfig());
        }

        public Task<bool> TextToImageFile(string prompt, string negativePrompt, string filename)
        {
            return TextToImageFileInternal(prompt, negativePrompt, filename, new SchedulerConfig());
        }

        public Task<bool> TextToImageFile(string prompt, string filename, SchedulerConfig schedulerConfig)
        {
            return TextToImageFileInternal(prompt, null, filename, schedulerConfig);
        }

        public Task<bool> TextToImageFile(string prompt, string negativePrompt, string filename, SchedulerConfig schedulerConfig)
        {
            return TextToImageFileInternal(prompt, negativePrompt, filename, schedulerConfig);
        }



        private async Task<Image<Rgba32>> TextToImageInternal(string prompt, string negativePrompt, SchedulerConfig schedulerConfig)
        {
            return await Task.Run(() =>
            {
                var imageTensorData = _inferenceService.RunInference(prompt, negativePrompt, schedulerConfig);
                return _imageService.TensorToImage(imageTensorData, _configuration.Width, _configuration.Height);
            }).ConfigureAwait(false);
        }

        private async Task<bool> TextToImageFileInternal(string prompt, string negativePrompt, string filename, SchedulerConfig schedulerConfig)
        {
            var image = await TextToImageInternal(prompt, negativePrompt, schedulerConfig);
            if (image is null)
                return false;

            await image.SaveAsync(filename);
            return true;
        }

        public void Dispose()
        {
            _inferenceService.Dispose();
        }


    }
}
