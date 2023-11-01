using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Enums;
using System.Collections.Concurrent;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IPipeline
    {

        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        DiffuserPipelineType PipelineType { get; }


        /// <summary>
        /// The pipelines diffuser set.
        /// </summary>
        ConcurrentDictionary<DiffuserType, IDiffuser> Diffusers { get; }


        /// <summary>
        /// Gets the diffuser.
        /// </summary>
        /// <param name="diffuserType">Type of the diffuser.</param>
        /// <returns></returns>
        IDiffuser GetDiffuser(DiffuserType diffuserType);
    }
}