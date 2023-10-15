using OnnxStack.WebUI.Models;

namespace Services
{
    public interface IFileService
    {
        /// <summary>
        /// Gets the input image file.
        /// </summary>
        /// <param name="imageName">Name of the image.</param>
        /// <returns></returns>
        Task<FileServiceResult> GetInputImageFile(string imageName);

        /// <summary>
        /// Saves the blueprint file.
        /// </summary>
        /// <param name="bluepring">The blueprint.</param>
        /// <param name="bluprintFile">The bluprint file.</param>
        /// <returns></returns>
        Task<FileServiceResult> SaveBlueprintFile(ImageBlueprint blueprint, string bluprintFile);

        /// <summary>
        /// Saves the image file.
        /// </summary>
        /// <param name="imageBase64">The image base64.</param>
        /// <param name="fileName">Name of the file.</param>
        /// <returns></returns>
        Task<FileServiceResult> SaveImageFile(string imageBase64, string fileName);

        /// <summary>
        /// Creates a random name.
        /// </summary>
        /// <returns></returns>
        Task<string> CreateRandomName();

        /// <summary>
        /// Convert URL to physical path.
        /// </summary>
        /// <param name="url">The URL.</param>
        /// <returns></returns>
        Task<string> UrlToPhysicalPath(string url);

        /// <summary>
        /// Creates the output URL.
        /// </summary>
        /// <param name="file">The file.</param>
        /// <returns></returns>
        Task<string> CreateOutputUrl(string file, bool relative = true);
    }
}