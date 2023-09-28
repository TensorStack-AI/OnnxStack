using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Results;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Services
{
    public sealed class StableDiffusionService : IStableDiffusionService
    {
        private readonly IInferenceService _inferenceService;

        public StableDiffusionService(IInferenceService inferenceService)
        {
            _inferenceService = inferenceService;
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
            var imageTensorData = await _inferenceService.RunInferenceAsync(prompt, options).ConfigureAwait(false);
            return TensorToImage(options, imageTensorData);
        }

        private async Task<ImageResult> TextToImageFileInternal(PromptOptions prompt, SchedulerOptions options, string outputFile)
        {
            var result = await TextToImageInternal(prompt, options);
            if (result is null)
                return null;

            await result.SaveAsync(outputFile);
            return result;
        }


        private ImageResult TensorToImage(SchedulerOptions options, DenseTensor<float> imageTensor)
        {
            var result = new Image<Rgba32>(options.Width, options.Height);
            for (var y = 0; y < options.Height; y++)
            {
                for (var x = 0; x < options.Width; x++)
                {
                    result[x, y] = new Rgba32(
                        CalculateByte(imageTensor, 0, y, x),
                        CalculateByte(imageTensor, 1, y, x),
                        CalculateByte(imageTensor, 2, y, x)
                    );
                }
            }
            return new ImageResult(result);
        }

        private static byte CalculateByte(Tensor<float> imageTensor, int index, int y, int x)
        {
            return (byte)Math.Round(Math.Clamp(imageTensor[0, index, y, x] / 2 + 0.5, 0, 1) * 255);
        }
    }
}
