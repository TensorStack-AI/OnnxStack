using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Diffusers.StableDiffusion;
using OnnxStack.StableDiffusion.Enums;
using System.Collections.Concurrent;
using System.Collections.Generic;

namespace OnnxStack.StableDiffusion.Pipelines
{
    public sealed class StableDiffusionPipeline : IPipeline
    {
        private readonly DiffuserPipelineType _pipelineType;
        private readonly ConcurrentDictionary<DiffuserType, IDiffuser> _diffusers;


        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionPipeline"/> class.
        /// </summary>
        /// <param name="onnxModelService">The onnx model service.</param>
        /// <param name="promptService">The prompt service.</param>
        public StableDiffusionPipeline(IOnnxModelService onnxModelService, IPromptService promptService)
        {
            var diffusers = new Dictionary<DiffuserType, IDiffuser>
            {
                { DiffuserType.TextToImage, new TextDiffuser(onnxModelService, promptService) },
                { DiffuserType.ImageToImage, new ImageDiffuser(onnxModelService, promptService) },
                { DiffuserType.ImageInpaint, new InpaintDiffuser(onnxModelService, promptService) },
                { DiffuserType.ImageInpaintLegacy, new InpaintLegacyDiffuser(onnxModelService, promptService) }
            };
            _pipelineType = DiffuserPipelineType.StableDiffusion;
            _diffusers = new ConcurrentDictionary<DiffuserType, IDiffuser>(diffusers);
        }


        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public DiffuserPipelineType PipelineType => _pipelineType;


        /// <summary>
        /// Gets the diffusers.
        /// </summary>
        public ConcurrentDictionary<DiffuserType, IDiffuser> Diffusers => _diffusers;


        /// <summary>
        /// Gets the diffuser.
        /// </summary>
        /// <param name="diffuserType">Type of the diffuser.</param>
        /// <returns></returns>
        public IDiffuser GetDiffuser(DiffuserType diffuserType)
        {
            _diffusers.TryGetValue(diffuserType, out var diffuser);
            return diffuser;
        }
    }
}
