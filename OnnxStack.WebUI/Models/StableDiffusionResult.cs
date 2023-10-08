namespace OnnxStack.WebUI.Models
{
    public sealed class StableDiffusionResult
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionResult"/> class.
        /// </summary>
        /// <param name="imageName">Name of the image.</param>
        /// <param name="imageUrl">The image URL.</param>
        /// <param name="blueprint">The blueprint.</param>
        /// <param name="blueprintName">Name of the blueprint.</param>
        /// <param name="blueprintUrl">The blueprint URL.</param>
        /// <param name="elapsed">The elapsed.</param>
        public StableDiffusionResult(string imageName, string imageUrl, ImageBlueprint blueprint, string blueprintName, string blueprintUrl, int elapsed)
        {
            ImageName = imageName;
            ImageUrl = imageUrl;
            Blueprint = blueprint;
            BlueprintName = blueprintName;
            BlueprintUrl = blueprintUrl;
            Elapsed = elapsed;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionResult"/> class.
        /// </summary>
        /// <param name="error">The error.</param>
        public StableDiffusionResult(string error)
        {
            Error = error;
        }

        public string ImageName { get; set; }
        public string ImageUrl { get; set; }
        public ImageBlueprint Blueprint { get; set; }
        public string BlueprintName { get; set; }
        public string BlueprintUrl { get; set; }
        public int Elapsed { get; set; }
        public string Error { get; set; }
        public bool IsError => !string.IsNullOrEmpty(Error);
    }
}
