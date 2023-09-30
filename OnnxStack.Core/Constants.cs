using System.Collections.Immutable;
using System.Linq;

namespace OnnxStack.Core
{
    public static class Constants
    {
        /// <summary>
        /// The blank token value
        /// </summary>
        public const int BlankTokenValue = 49407;

        /// <summary>
        /// The maximum input token count
        /// </summary>
        public const int MaxInputTokenCount = 2048;

        /// <summary>
        /// The clip tokenizer input token limit
        /// </summary>
        public const int ClipTokenizerTokenLimit = 77;

        /// <summary>
        /// The clip tokenizer embeddings length
        /// </summary>
        public const int ClipTokenizerEmbeddingsLength = 768;

        /// <summary>
        /// The cached blank token value array
        /// </summary>
        public static ImmutableArray<int> BlankTokenValueArray;

        /// <summary>
        /// The model scale factor
        /// </summary>
        public const float ModelScaleFactor = 0.18215f;

        static Constants()
        {
            // Cache an array with enough blank tokens to fill an empty prompt
            BlankTokenValueArray = Enumerable.Repeat(BlankTokenValue, MaxInputTokenCount).ToImmutableArray();
        }
    }
}
