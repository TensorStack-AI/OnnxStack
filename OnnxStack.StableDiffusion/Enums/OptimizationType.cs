namespace OnnxStack.StableDiffusion.Enums
{
    public enum OptimizationType
    {
        /// <summary>
        ///No optimizations
        /// </summary>
        None = 0,

        /// <summary>
        /// Level1 - ONNX Graph optimizations
        /// </summary>
        Level1 = 1,

        /// <summary>
        /// Level2 - Level1 + Batch + Channel optimizations
        /// </summary>
        Level2 = 2,

        /// <summary>
        /// Level3 - Level2 + Width + Height optimizations
        /// Note: Will reload session if Width or Height has changed
        /// </summary>
        Level3 = 3,

        /// <summary>
        /// Level4 - Level3 + Prompt optimizations
        /// Note: Will reload session if Width, Height or Prompt token length has changed
        /// </summary>
        Level4 = 4 
    }
}
