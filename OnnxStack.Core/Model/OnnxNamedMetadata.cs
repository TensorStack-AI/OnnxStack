using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;

namespace OnnxStack.Core.Model
{
    public record OnnxNamedMetadata(string Name, NodeMetadata Value)
    {
        internal static OnnxNamedMetadata Create(KeyValuePair<string, NodeMetadata> metadata)
        {
            return new OnnxNamedMetadata(metadata.Key, metadata.Value);
        }
    }
}
