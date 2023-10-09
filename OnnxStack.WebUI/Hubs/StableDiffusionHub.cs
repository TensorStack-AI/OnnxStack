using Microsoft.AspNetCore.SignalR;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.WebUI.Models;
using Services;
using SixLabors.ImageSharp;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace OnnxStack.Web.Hubs
{
    public class StableDiffusionHub : Hub<IStableDiffusionClient>
    {
        private readonly IFileService _fileService;
        private readonly ILogger<StableDiffusionHub> _logger;
        private readonly IStableDiffusionService _stableDiffusionService;

        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionHub"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="stableDiffusionService">The stable diffusion service.</param>
        /// <param name="webHostEnvironment">The web host environment.</param>
        public StableDiffusionHub(ILogger<StableDiffusionHub> logger, IStableDiffusionService stableDiffusionService, IFileService fileService)
        {
            _logger = logger;
            _fileService = fileService;
            _stableDiffusionService = stableDiffusionService;
        }


        /// <summary>
        /// Called when a new connection is established with the hub.
        /// </summary>
        public override async Task OnConnectedAsync()
        {
            _logger.Log(LogLevel.Information, "[OnConnectedAsync], Id: {0}", Context.ConnectionId);
            await Clients.Caller.OnMessage("OnConnectedAsync");
            await base.OnConnectedAsync();
        }


        /// <summary>
        /// Called when a connection with the hub is terminated.
        /// </summary>
        /// <param name="exception"></param>
        public override async Task OnDisconnectedAsync(Exception exception)
        {
            _logger.Log(LogLevel.Information, "[OnDisconnectedAsync], Id: {0}", Context.ConnectionId);
            await Clients.Caller.OnMessage("OnDisconnectedAsync");
            await base.OnDisconnectedAsync(exception);
        }


        /// <summary>
        /// Execute Text-To-Image Stable Diffusion
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        [HubMethodName("ExecuteTextToImage")]
        public async IAsyncEnumerable<StableDiffusionResult> OnExecuteTextToImage(PromptOptions promptOptions, SchedulerOptions schedulerOptions, [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            _logger.Log(LogLevel.Information, "[OnExecuteTextToImage] - New request received, Connection: {0}", Context.ConnectionId);
            var cancellationTokenSource = CancellationTokenSource.CreateLinkedTokenSource(Context.ConnectionAborted, cancellationToken);

            // TODO: Add support for multiple results
            var result = await GenerateTextToImageResult(promptOptions, schedulerOptions, cancellationTokenSource.Token);
            if (!result.IsError)
                yield return result;

            await Clients.Caller.OnError(result.Error);
        }


        /// <summary>
        /// Execute Image-To-Image Stable Diffusion
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        [HubMethodName("ExecuteImageToImage")]
        public async IAsyncEnumerable<StableDiffusionResult> OnExecuteImageToImage(PromptOptions promptOptions, SchedulerOptions schedulerOptions, [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            _logger.Log(LogLevel.Information, "[ExecuteImageToImage] - New request received, Connection: {0}", Context.ConnectionId);
            var cancellationTokenSource = CancellationTokenSource.CreateLinkedTokenSource(Context.ConnectionAborted, cancellationToken);

            // TODO: Add support for multiple results
            var result = await GenerateImageToImageResult(promptOptions, schedulerOptions, cancellationTokenSource.Token);
            if (!result.IsError)
                yield return result;

            await Clients.Caller.OnError(result.Error);
            yield break;
        }


        /// <summary>
        /// Generates the image to image result.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        private async Task<StableDiffusionResult> GenerateImageToImageResult(PromptOptions promptOptions, SchedulerOptions schedulerOptions, CancellationToken cancellationToken)
        {
            var timestamp = Stopwatch.GetTimestamp();
            schedulerOptions.Seed = GenerateSeed(schedulerOptions.Seed);

            //1. Create filenames
            var random = await _fileService.CreateRandomName();
            var output = $"Output-{random}";
            var outputImage = $"{output}.png";
            var outputBlueprint = $"{output}.json";
            var inputImage = $"Input-{random}.png";
            var outputImageUrl = await _fileService.CreateOutputUrl(outputImage);
            var outputImageFile = await _fileService.UrlToPhysicalPath(outputImageUrl);
            var inputOriginaUrl = await _fileService.CreateOutputUrl(promptOptions.InputImage.ImagePath);

            //2. Copy input image to new file
            var inputImageFile = await _fileService.CopyInputImageFile(promptOptions.InputImage.ImagePath, inputImage);
            if (inputImageFile is null)
                return new StableDiffusionResult("Failed to copy input image");

            //3. Generate blueprint
            var inputImageLink = await _fileService.CreateOutputUrl(inputImage, false);
            var outputImageLink = await _fileService.CreateOutputUrl(outputImage, false);
            promptOptions.InputImage = new InputImage(inputOriginaUrl);
            var blueprint = new ImageBlueprint(promptOptions, schedulerOptions, outputImageLink, inputImageLink);
            var bluprintFile = await _fileService.SaveBlueprintFile(blueprint, outputBlueprint);
            if (bluprintFile is null)
                return new StableDiffusionResult("Failed to save blueprint");

            //4. Set full path of input image
            promptOptions.InputImage = new InputImage(inputImageFile.FilePath);

            //5. Run stable diffusion
            if (!await RunStableDiffusion(promptOptions, schedulerOptions, outputImageFile, cancellationToken))
                return new StableDiffusionResult("Failed to run stable diffusion");

            //6. Return result
            return new StableDiffusionResult(outputImage, outputImageUrl, blueprint, bluprintFile.Filename, bluprintFile.FileUrl, GetElapsed(timestamp));
        }


        /// <summary>
        /// Generates the text to image result.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        private async Task<StableDiffusionResult> GenerateTextToImageResult(PromptOptions promptOptions, SchedulerOptions schedulerOptions, CancellationToken cancellationToken)
        {
            var timestamp = Stopwatch.GetTimestamp();
            schedulerOptions.Seed = GenerateSeed(schedulerOptions.Seed);

            //1. Create filenames
            var random = await _fileService.CreateRandomName();
            var output = $"Output-{random}";
            var outputImage = $"{output}.png";
            var outputBlueprint = $"{output}.json";
            var outputImageUrl = await _fileService.CreateOutputUrl(outputImage);
            var outputImageFile = await _fileService.UrlToPhysicalPath(outputImageUrl);

            //2. Generate blueprint
            var outputImageLink = await _fileService.CreateOutputUrl(outputImage, false);
            var blueprint = new ImageBlueprint(promptOptions, schedulerOptions, outputImageLink);
            var bluprintFile = await _fileService.SaveBlueprintFile(blueprint, outputBlueprint);
            if (bluprintFile is null)
                return new StableDiffusionResult("Failed to save blueprint");

            //3. Run stable diffusion
            if (!await RunStableDiffusion(promptOptions, schedulerOptions, outputImageFile, cancellationToken))
                return new StableDiffusionResult("Failed to run stable diffusion");

            //4. Return result
            return new StableDiffusionResult(outputImage, outputImageUrl, blueprint, bluprintFile.Filename, bluprintFile.FileUrl, GetElapsed(timestamp));
        }


        /// <summary>
        /// Runs the stable diffusion.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="fileInfo">The file information.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        private async Task<bool> RunStableDiffusion(PromptOptions promptOptions, SchedulerOptions schedulerOptions, string outputImage, CancellationToken cancellationToken)
        {
            try
            {
                var resultImage = await _stableDiffusionService.GenerateAsImageAsync(promptOptions, schedulerOptions, ProgressCallback(), cancellationToken);
                if (resultImage is null)
                    return false;

                await resultImage.SaveAsPngAsync(outputImage).ConfigureAwait(false);
                return true;
            }
            catch (OperationCanceledException tex)
            {
                await Clients.Caller.OnCanceled(tex.Message);
                _logger.Log(LogLevel.Warning, tex, "[RunStableDiffusion] - Operation canceled, Connection: {0}", Context.ConnectionId);
            }
            catch (Exception ex)
            {
                await Clients.Caller.OnError(ex.Message);
                _logger.Log(LogLevel.Error, ex, "[RunStableDiffusion] - Error generating image, Connection: {0}", Context.ConnectionId);
            }
            return false;
        }


        /// <summary>
        /// Progress callback.
        /// </summary>
        /// <returns></returns>
        private Action<int, int> ProgressCallback()
        {
            return async (progress, total) =>
            {
                _logger.Log(LogLevel.Information, "[ProgressCallback] - Progress: {0}/{1}, Connection: {2}", progress, total, Context.ConnectionId);
                await Clients.Caller.OnProgress(new ProgressResult(progress, total));
            };
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
        /// Gets the elapsed time is seconds.
        /// </summary>
        /// <param name="timestamp">The begin timestamp.</param>
        /// <returns></returns>
        private static int GetElapsed(long timestamp)
        {
            return (int)Stopwatch.GetElapsedTime(timestamp).TotalSeconds;
        }
    }
}
