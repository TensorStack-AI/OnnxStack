using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using System;
using System.Threading;
using System.Threading.Tasks;


namespace OnnxStack.StableDiffusion.Services
{
    public sealed class DiffuserService : IDiffuserService
    {
        private readonly IDiffuser _textDiffuser;
        private readonly IDiffuser _imageDiffuser;
        private readonly IDiffuser _inpaintDiffuser;
        private readonly IDiffuser _inpaintLegacyDiffuser;

        /// <summary>
        /// Initializes a new instance of the <see cref="DiffuserService"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="onnxModelService">The onnx model service.</param>
        public DiffuserService(IOnnxModelService onnxModelService, IPromptService promptService)
        {
            _textDiffuser = new TextDiffuser(onnxModelService, promptService);
            _imageDiffuser = new ImageDiffuser(onnxModelService, promptService);
            _inpaintDiffuser = new InpaintDiffuser(onnxModelService, promptService);
            _inpaintLegacyDiffuser = new InpaintLegacyDiffuser(onnxModelService, promptService);
        }


        /// <summary>
        /// Runs the Stable Diffusion inference.
        /// </summary>
        /// <param name="promptOptions">The options.</param>
        /// <param name="schedulerOptions">The scheduler configuration.</param>
        /// <returns></returns>
        public async Task<DenseTensor<float>> RunAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions, Action<int, int> progress = null, CancellationToken cancellationToken = default)
        {
            return promptOptions.ProcessType switch
            {
                ProcessType.TextToImage => await _textDiffuser.DiffuseAsync(promptOptions, schedulerOptions, progress, cancellationToken),
                ProcessType.ImageToImage => await _imageDiffuser.DiffuseAsync(promptOptions, schedulerOptions, progress, cancellationToken),
                ProcessType.ImageInpaint => await _inpaintLegacyDiffuser.DiffuseAsync(promptOptions, schedulerOptions, progress, cancellationToken),
                _ => throw new NotImplementedException()
            };
        }
    }
}
