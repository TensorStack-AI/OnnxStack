using Microsoft.AspNetCore.Hosting.Server.Features;
using OnnxStack.WebUI.Models;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Services
{
    public class FileService : IFileService
    {
        private readonly ILogger<FileService> _logger;
        private readonly IWebHostEnvironment _webHostEnvironment;
        private readonly JsonSerializerOptions _serializerOptions;
        private readonly IServerAddressesFeature _serverAddressesFeature;

        /// <summary>
        /// Initializes a new instance of the <see cref="FileService"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="webHostEnvironment">The web host environment.</param>
        public FileService(ILogger<FileService> logger, IWebHostEnvironment webHostEnvironment, IServerAddressesFeature serverAddressesFeature)
        {
            _logger = logger;
            _webHostEnvironment = webHostEnvironment;
            _serverAddressesFeature = serverAddressesFeature;
            _serializerOptions = new JsonSerializerOptions { WriteIndented = true, Converters = { new JsonStringEnumConverter() } };
        }


        /// <summary>
        /// Gets the input image file.
        /// </summary>
        /// <param name="imageName">Name of the image.</param>
        /// <returns></returns>
        public async Task<FileServiceResult> GetInputImageFile(string imageName)
        {
            try
            {
                if (string.IsNullOrEmpty(imageName))
                    return null;

                var outputImageUrl = await CreateOutputUrl(imageName);
                var outputImageFile = await UrlToPhysicalPath(outputImageUrl);

                if (!File.Exists(outputImageFile))
                    return null;

                return new FileServiceResult(imageName, outputImageUrl, outputImageFile);
            }
            catch (Exception ex)
            {
                _logger.Log(LogLevel.Error, ex, "[GetInputImageFile] - Error getting image file");
                return null;
            }
        }


        /// <summary>
        /// Saves the blueprint file.
        /// </summary>
        /// <param name="bluprint">The bluprint.</param>
        /// <param name="fileName">Name of the file.</param>
        /// <returns></returns>
        public async Task<FileServiceResult> SaveBlueprintFile(ImageBlueprint bluprint, string fileName)
        {
            try
            {
                var outputImageUrl = await CreateOutputUrl(fileName);
                var outputImageFile = await UrlToPhysicalPath(outputImageUrl);
                using (var stream = File.Create(outputImageFile))
                {
                    await JsonSerializer.SerializeAsync(stream, bluprint, _serializerOptions);
                    return new FileServiceResult(fileName, outputImageUrl, outputImageFile);
                }
            }
            catch (Exception ex)
            {
                _logger.Log(LogLevel.Error, ex, "[SaveBlueprintFile] - Error saving Blueprint file");
                return null;
            }
        }


        /// <summary>
        /// Saves the image file.
        /// </summary>
        /// <param name="imageBase64">The image base64.</param>
        /// <param name="fileName">Name of the file.</param>
        /// <returns></returns>
        public async Task<FileServiceResult> SaveImageFile(string imageBase64, string fileName)
        {
            try
            {
                var outputImageUrl = await CreateOutputUrl(fileName);
                var outputImageFile = await UrlToPhysicalPath(outputImageUrl);
                await File.WriteAllBytesAsync(outputImageFile, Convert.FromBase64String(imageBase64.Split(',')[1]));
                return new FileServiceResult(fileName, outputImageUrl, outputImageFile);
            }
            catch (Exception ex)
            {
                _logger.Log(LogLevel.Error, ex, "[SaveImageFile] - Error saving imageBase64 file");
                return null;
            }
        }


        /// <summary>
        /// URL path to physical path.
        /// </summary>
        /// <param name="url">The URL.</param>
        /// <returns></returns>
        public Task<string> UrlToPhysicalPath(string url)
        {
            string webRootPath = _webHostEnvironment.WebRootPath;
            string sanitizedUrl = url.Replace('/', Path.DirectorySeparatorChar);
            string physicalPath = Path.Combine(webRootPath, sanitizedUrl.TrimStart(Path.DirectorySeparatorChar));
            return Task.FromResult(physicalPath);
        }


        /// <summary>
        /// Creates the output URL.
        /// </summary>
        /// <param name="folder">The folder.</param>
        /// <param name="file">The file.</param>
        /// <returns></returns>
        public Task<string> CreateOutputUrl(string file, bool relative = true)
        {
            return relative
                ? Task.FromResult($"/images/results/{file}")
                : Task.FromResult($"{GetServerUrl()}/images/results/{file}");
        }


        /// <summary>
        /// Creates a random name.
        /// </summary>
        /// <returns></returns>
        public Task<string> CreateRandomName()
        {
            return Task.FromResult(Path.GetFileNameWithoutExtension(Path.GetRandomFileName()));
        }


        /// <summary>
        /// Gets the server URL.
        /// </summary>
        /// <returns></returns>
        private string GetServerUrl()
        {
            var address = _serverAddressesFeature.Addresses.FirstOrDefault(x => x.StartsWith("https"))
                       ?? _serverAddressesFeature.Addresses.FirstOrDefault();
            if (string.IsNullOrEmpty(address))
                return string.Empty;

            return address;
        }
    }

    public record FileServiceResult(string Filename, string FileUrl, string FilePath);
}
