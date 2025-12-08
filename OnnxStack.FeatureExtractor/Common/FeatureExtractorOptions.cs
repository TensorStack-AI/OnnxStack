using OnnxStack.Core.Image;

namespace OnnxStack.FeatureExtractor.Common
{
    public record FeatureExtractorOptions
    {
        public FeatureExtractorOptions(TileMode tileMode = TileMode.None, int maxTileSize = 512, int tileOverlap = 16, bool isLowMemoryEnabled = false)
        {
            TileMode = tileMode;
            MaxTileSize = maxTileSize;
            TileOverlap = tileOverlap;
            IsLowMemoryEnabled = isLowMemoryEnabled;
        }


        /// <summary>
        /// Gets or sets the tile mode.
        /// </summary>
        public TileMode TileMode { get; set; }

        /// <summary>
        /// The maximum size of the tile.
        /// </summary>
        public int MaxTileSize { get; }

        /// <summary>
        /// The tile overlap in pixels to avoid visible seams.
        /// </summary>
        public int TileOverlap { get; }

        /// <summary>
        /// Gets or sets a value indicating whether this instance is low memory enabled.
        /// </summary>
        public bool IsLowMemoryEnabled { get; set; }

        /// <summary>
        /// Gets or sets the value.
        /// </summary>
        public float Value { get; set; }
    }
}
