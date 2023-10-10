using System.Collections.Generic;

namespace OnnxStack.Core
{
    public static class Constants
    {
        /// <summary>
        /// The width/height valid sizes
        /// </summary>
        public static readonly IReadOnlyList<int> ValidSizes;

        static Constants()
        {
            // Cache an array with enough blank tokens to fill an empty prompt
            ValidSizes = new List<int> { 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024 };
        }
    }
}
