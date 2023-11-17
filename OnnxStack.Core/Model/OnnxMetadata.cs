using System.Collections.Generic;

namespace OnnxStack.Core.Model
{
    public record OnnxMetadata
    {
        /// <summary>
        /// Gets or sets the inputs.
        /// </summary>
        public IReadOnlyList<OnnxNamedMetadata> Inputs { get; set; }

        /// <summary>
        /// Gets or sets the outputs.
        /// </summary>
        public IReadOnlyList<OnnxNamedMetadata> Outputs { get; set; }
    }
}
