using OnnxStack.Web.Models;
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
        /// Copies the input image file.
        /// </summary>
        /// <param name="sourceImage">The source image.</param>
        /// <param name="destinationImage">The destination image.</param>
        /// <returns></returns>
        Task<FileServiceResult> CopyInputImageFile(string sourceImage, string destinationImage);

        /// <summary>
        /// Uploads the image file.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        Task<FileServiceResult> UploadImageFile(UploadImageModel model);

        /// <summary>
        /// Saves the blueprint file.
        /// </summary>
        /// <param name="bluepring">The blueprint.</param>
        /// <param name="bluprintFile">The bluprint file.</param>
        /// <returns></returns>
        Task<FileServiceResult> SaveBlueprintFile(ImageBlueprint blueprint, string bluprintFile);

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
        Task<string> CreateOutputUrl(string file);
    }
}