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

        public Task<Image<Rgba32>> TextToImage(StableDiffusionOptions options)
        {
            return TextToImageInternal(options, new SchedulerOptions());
        }

        public Task<Image<Rgba32>> TextToImage(StableDiffusionOptions options, SchedulerOptions schedulerOptions)
        {
            return TextToImageInternal(options, schedulerOptions);
        }

        public Task<bool> TextToImageFile(StableDiffusionOptions options, string outputFile)
        {
            return TextToImageFileInternal(options, new SchedulerOptions(), outputFile);
        }

        public Task<bool> TextToImageFile(StableDiffusionOptions options, SchedulerOptions schedulerOptions, string outputFile)
        {
            return TextToImageFileInternal(options, schedulerOptions, outputFile);
        }


        private async Task<Image<Rgba32>> TextToImageInternal(StableDiffusionOptions options, SchedulerOptions schedulerConfig)
        {
            return await Task.Run(() =>
            {
                var imageTensorData = _inferenceService.RunInference(options, schedulerConfig);
                return _imageService.TensorToImage(options, imageTensorData);
            }).ConfigureAwait(false);
        }

        private async Task<bool> TextToImageFileInternal(StableDiffusionOptions options, SchedulerOptions schedulerConfig, string outputFile)
        {
            var image = await TextToImageInternal(options, schedulerConfig);
            if (image is null)
                return false;

            await image.SaveAsync(outputFile);
            return true;
        }

        public void Dispose()
        {
            _inferenceService.Dispose();
        }
    }
}
