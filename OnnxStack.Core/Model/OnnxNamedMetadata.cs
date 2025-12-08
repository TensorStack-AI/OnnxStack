using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;

namespace OnnxStack.Core.Model
{
    public sealed record OnnxNamedMetadata(string Name, NodeMetadata Value)
    {
        public ReadOnlySpan<int> Dimensions => Value.Dimensions;

        internal static OnnxNamedMetadata Create(KeyValuePair<string, NodeMetadata> metadata)
        {
            return new OnnxNamedMetadata(metadata.Key, metadata.Value);
        }
    }
}
