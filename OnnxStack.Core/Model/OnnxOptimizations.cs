using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxStack.Core.Model
{
    public record OnnxOptimizations : IEquatable<OnnxOptimizations>
    {
        private readonly GraphOptimizationLevel _graphOptimizationLevel;
        private readonly SortedDictionary<string, long> _dimensionOverrides;

        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxOptimizations"/> class.
        /// </summary>
        public OnnxOptimizations() : this(GraphOptimizationLevel.ORT_ENABLE_ALL) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxOptimizations"/> class.
        /// </summary>
        /// <param name="freeDimensionOverrides">The free dimension overrides.</param>
        public OnnxOptimizations(GraphOptimizationLevel graphOptimization)
        {
            _graphOptimizationLevel = graphOptimization;
            _dimensionOverrides = new SortedDictionary<string, long>();
        }


        /// <summary>
        /// Gets the graph optimization level.
        /// </summary>
        public GraphOptimizationLevel GraphOptimizationLevel => _graphOptimizationLevel;


        /// <summary>
        /// Gets the dimension overrides.
        /// </summary>
        public SortedDictionary<string, long> DimensionOverrides => _dimensionOverrides;


        /// <summary>
        /// Adds the specified dimension override.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <param name="value">The value.</param>
        public void Add(string dimension, long value)
        {
            _dimensionOverrides.Add(dimension, value);
        }


        /// <summary>
        /// Removes the specified dimensions.
        /// </summary>
        /// <param name="dimensions">The dimensions.</param>
        public void Remove(params string[] dimensions)
        {
            foreach (var dimension in dimensions)
            {
                _dimensionOverrides.Remove(dimension);
            }
        }


        /// <summary>
        // Indicates whether the current OnnxOptimizations is equal to another
        /// </summary>
        /// <param name="other">The other.</param>
        /// <returns><c>true</c> if equal, <c>false</c> otherwise.</returns>
        public virtual bool Equals(OnnxOptimizations other)
        {
            if (other is null)
                return false;

            return other.DimensionOverrides.SequenceEqual(DimensionOverrides);
        }


        /// <summary>
        /// Returns a hash code for this instance.
        /// </summary>
        /// <returns>A hash code for this instance, suitable for use in hashing algorithms and data structures like a hash table.</returns>
        public override int GetHashCode()
        {
            return HashCode.Combine(DimensionOverrides.GetHashCode());
        }
    }
}
