using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Diffusers.LatentConsistency;
using OnnxStack.StableDiffusion.Enums;
using System.Collections.Concurrent;
using System.Collections.Generic;

namespace OnnxStack.StableDiffusion.Pipelines
{
    public sealed class LatentConsistencyPipeline : IPipeline
    {
        private readonly DiffuserPipelineType _pipelineType;
        private readonly ConcurrentDictionary<DiffuserType, IDiffuser> _diffusers;

        public LatentConsistencyPipeline(IOnnxModelService onnxModelService, IPromptService promptService)
        {
            var diffusers = new Dictionary<DiffuserType, IDiffuser>
            {
               { DiffuserType.TextToImage, new TextDiffuser(onnxModelService, promptService) },
               { DiffuserType.ImageToImage, new ImageDiffuser(onnxModelService, promptService) }
            };
            _pipelineType = DiffuserPipelineType.LatentConsistency;
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
