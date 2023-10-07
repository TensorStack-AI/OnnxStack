using Microsoft.AspNetCore.SignalR;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.WebUI.Models;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace OnnxStack.Web.Hubs
{
    public class StableDiffusionHub : Hub
    {
        private readonly ILogger<StableDiffusionHub> _logger;
        private readonly IWebHostEnvironment _webHostEnvironment;
        private readonly JsonSerializerOptions _serializerOptions;
        private readonly IStableDiffusionService _stableDiffusionService;


        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionHub"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="stableDiffusionService">The stable diffusion service.</param>
        /// <param name="webHostEnvironment">The web host environment.</param>
        public StableDiffusionHub(ILogger<StableDiffusionHub> logger, IStableDiffusionService stableDiffusionService, IWebHostEnvironment webHostEnvironment)
        {
            _logger = logger;
            _webHostEnvironment = webHostEnvironment;
            _stableDiffusionService = stableDiffusionService;
            _serializerOptions = new JsonSerializerOptions { WriteIndented = true, Converters = { new JsonStringEnumConverter() } };
        }


        /// <summary>
        /// Called when a new connection is established with the hub.
        /// </summary>
        public override async Task OnConnectedAsync()
        {
            _logger.Log(LogLevel.Information, "[OnConnectedAsync], Id: {0}", Context.ConnectionId);
            await Clients.Caller.SendAsync("OnMessage", "OnConnectedAsync");
            await base.OnConnectedAsync();
        }


        /// <summary>
        /// Called when a connection with the hub is terminated.
        /// </summary>
        /// <param name="exception"></param>
        public override async Task OnDisconnectedAsync(Exception exception)
        {
            _logger.Log(LogLevel.Information, "[OnDisconnectedAsync], Id: {0}", Context.ConnectionId);
            await Clients.Caller.SendAsync("OnMessage", "OnDisconnectedAsync");
            await base.OnDisconnectedAsync(exception);
        }


        /// <summary>
        /// Execute Text-To-Image Stable Diffusion
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        [HubMethodName("ExecuteTextToImage")]
        public async IAsyncEnumerable<TextToImageResult> OnExecuteTextToImage(PromptOptions promptOptions, SchedulerOptions schedulerOptions, [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            _logger.Log(LogLevel.Information, "[OnExecuteTextToImage] - New request received, Connection: {0}", Context.ConnectionId);
            var cancellationTokenSource = CancellationTokenSource.CreateLinkedTokenSource(Context.ConnectionAborted, cancellationToken);

            // TODO: Add support for multiple results
            var result = await GenerateTextToImageResult(promptOptions, schedulerOptions, cancellationTokenSource.Token);
            if (result is null)
                yield break;

            yield return result;
        }


        /// <summary>
        /// Generates the text to image result.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        private async Task<TextToImageResult> GenerateTextToImageResult(PromptOptions promptOptions, SchedulerOptions schedulerOptions, CancellationToken cancellationToken)
        {
            var timestamp = Stopwatch.GetTimestamp();
            schedulerOptions.Seed = GenerateSeed(schedulerOptions.Seed);

            var blueprint = new ImageBlueprint(promptOptions, schedulerOptions);
            var fileInfo = CreateFileInfo(promptOptions, schedulerOptions);
            if (!await SaveBlueprintFile(fileInfo, blueprint))
                return null;

            if (!await RunStableDiffusion(promptOptions, schedulerOptions, fileInfo, cancellationToken))
                return null;

            var elapsed = (int)Stopwatch.GetElapsedTime(timestamp).TotalSeconds;
            return new TextToImageResult(fileInfo.Image, fileInfo.ImageUrl, blueprint, fileInfo.Blueprint, fileInfo.BlueprintUrl, elapsed);
        }


        /// <summary>
        /// Runs the stable diffusion.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="fileInfo">The file information.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        private async Task<bool> RunStableDiffusion(PromptOptions promptOptions, SchedulerOptions schedulerOptions, FileInfoResult fileInfo, CancellationToken cancellationToken)
        {
            try
            {
                await _stableDiffusionService.TextToImageFile(promptOptions, schedulerOptions, fileInfo.ImageFile, ProgressCallback(), cancellationToken);
                return true;
            }
            catch (OperationCanceledException tex)
            {
                await Clients.Caller.SendAsync("OnCanceled", tex.Message);
                _logger.Log(LogLevel.Warning, tex, "[OnExecuteTextToImage] - Operation canceled, Connection: {0}", Context.ConnectionId);
            }
            catch (Exception ex)
            {
                await Clients.Caller.SendAsync("OnError", ex.Message);
                _logger.Log(LogLevel.Error, ex, "[OnExecuteTextToImage] - Error generating image, Connection: {0}", Context.ConnectionId);
            }
            return false;
        }


        /// <summary>
        /// Saves the options file.
        /// </summary>
        /// <param name="fileInfo">The file information.</param>
        /// <param name="options">The options.</param>
        /// <returns></returns>
        private async Task<bool> SaveBlueprintFile(FileInfoResult fileInfo, ImageBlueprint bluprint)
        {
            try
            {
                using (var stream = File.Create(fileInfo.BlueprintFile))
                {
                    await JsonSerializer.SerializeAsync(stream, bluprint, _serializerOptions);
                    return true;
                }
            }
            catch (Exception ex)
            {
                _logger.Log(LogLevel.Error, ex, "[SaveOptions] - Error saving model card, Connection: {0}", Context.ConnectionId);
                return false;
            }
        }


        /// <summary>
        /// Creates the file information.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <returns></returns>
        private FileInfoResult CreateFileInfo(PromptOptions promptOptions, SchedulerOptions schedulerOptions)
        {
            var rand = Path.GetFileNameWithoutExtension(Path.GetRandomFileName());
            var output = $"{schedulerOptions.Seed}-{rand}";
            var outputImage = $"{output}.png";
            var outputImageUrl = CreateOutputUrl("TextToImage", outputImage);
            var outputImageFile = UrlToPhysicalPath(outputImageUrl);

            var outputJson = $"{output}.json";
            var outputJsonUrl = CreateOutputUrl("TextToImage", outputJson);
            var outputJsonFile = UrlToPhysicalPath(outputJsonUrl);
            return new FileInfoResult(outputImage, outputImageUrl, outputImageFile, outputJson, outputJsonUrl, outputJsonFile);
        }


        /// <summary>
        /// Generates the seed.
        /// </summary>
        /// <param name="seed">The seed.</param>
        /// <returns></returns>
        private int GenerateSeed(int seed)
        {
            if (seed > 0)
                return seed;

            return Random.Shared.Next();
        }


        /// <summary>
        /// Progress callback.
        /// </summary>
        /// <returns></returns>
        private Action<int, int> ProgressCallback()
        {
            return async (progress, total) =>
            {
                _logger.Log(LogLevel.Information, "[OnExecuteTextToImage] - Progress: {0}/{1}, Connection: {2}", progress, total, Context.ConnectionId);
                await Clients.Caller.SendAsync("OnProgress", new ProgressResult(progress, total));
            };
        }


        /// <summary>
        /// URL path to physical path.
        /// </summary>
        /// <param name="url">The URL.</param>
        /// <returns></returns>
        private string UrlToPhysicalPath(string url)
        {
            string webRootPath = _webHostEnvironment.WebRootPath;
            string physicalPath = Path.Combine(webRootPath, url.TrimStart('/').Replace('/', '\\'));
            return physicalPath;
        }


        /// <summary>
        /// Creates the output URL.
        /// </summary>
        /// <param name="folder">The folder.</param>
        /// <param name="file">The file.</param>
        /// <returns></returns>
        private string CreateOutputUrl(string folder, string file)
        {
            return $"/images/results/{folder}/{file}";
        }
    }
}
