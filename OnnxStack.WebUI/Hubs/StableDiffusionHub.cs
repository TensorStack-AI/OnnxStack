using Microsoft.AspNetCore.SignalR;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.WebUI.Models;
using System;
using System.Runtime.CompilerServices;

namespace OnnxStack.Web.Hubs
{
    public class StableDiffusionHub : Hub
    {
        private readonly ILogger<StableDiffusionHub> _logger;
        private readonly IStableDiffusionService _stableDiffusionService;
        private readonly IWebHostEnvironment _webHostEnvironment;
        public StableDiffusionHub(ILogger<StableDiffusionHub> logger, IStableDiffusionService stableDiffusionService, IWebHostEnvironment webHostEnvironment)
        {
            _logger = logger;
            _webHostEnvironment = webHostEnvironment;
            _stableDiffusionService = stableDiffusionService;
        }

        public override async Task OnConnectedAsync()
        {
            _logger.Log(LogLevel.Information, "[OnConnectedAsync], Id: {0}", Context.ConnectionId);

            await Clients.Caller.SendAsync("OnMessage", "OnConnectedAsync");
            await base.OnConnectedAsync();
        }


        public override async Task OnDisconnectedAsync(Exception exception)
        {
            _logger.Log(LogLevel.Information, "[OnDisconnectedAsync], Id: {0}", Context.ConnectionId);

            await Clients.Caller.SendAsync("OnMessage", "OnDisconnectedAsync");
            await base.OnDisconnectedAsync(exception);
        }


        [HubMethodName("ExecuteTextToImage")]
        public async IAsyncEnumerable<DiffusionResult> OnExecuteTextToImage(TextToImageOptions options, [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            _logger.Log(LogLevel.Information, "[OnExecuteTextToImage] - New prompt received, Connection: {0}", Context.ConnectionId);
            var linkedCancellationToken = CancellationTokenSource.CreateLinkedTokenSource(Context.ConnectionAborted, cancellationToken);

            var promptOptions = new PromptOptions
            {
                Prompt = options.Prompt,
                NegativePrompt = options.NegativePrompt,
                SchedulerType = options.SchedulerType
            };

            var schedulerOptions = new SchedulerOptions
            {
                Width = options.Width,
                Height = options.Height,
                Seed = GenerateSeed(options.Seed),
                InferenceSteps = options.InferenceSteps,
                GuidanceScale = options.GuidanceScale,
                Strength = options.Strength,
                InitialNoiseLevel = options.InitialNoiseLevel
            };

            // TODO: Add support for multiple results
            var result = await GenerateTextToImage(promptOptions, schedulerOptions, cancellationToken);
            if(result is null)
                yield break;

            yield return result;
        }

        private async Task<DiffusionResult> GenerateTextToImage(PromptOptions promptOptions, SchedulerOptions schedulerOptions, CancellationToken cancellationToken)
        {
            var rand = Path.GetFileNameWithoutExtension(Path.GetRandomFileName());
            var outputImage = $"{schedulerOptions.Seed}_{promptOptions.SchedulerType}_{rand}.png";
            var outputImageUrl = CreateOutputImageUrl("TextToImage", outputImage);
            var outputImageFile = CreateOutputImageFile(outputImageUrl);

            try
            {
                await _stableDiffusionService.TextToImageFile(promptOptions, schedulerOptions, outputImageFile, ProgressCallback(), cancellationToken);
                return new DiffusionResult(outputImage, outputImageUrl);
            }
            catch (OperationCanceledException tex)
            {
                await Clients.Caller.SendAsync("OnCanceled", tex.Message);
            }
            catch (Exception ex)
            {
                await Clients.Caller.SendAsync("OnError", ex.Message);
            }
            return null;
        }

        private int GenerateSeed(int seed)
        {
            if (seed > 0)
                return seed;

            return Random.Shared.Next();
        }

        private Action<int, int> ProgressCallback()
        {
            return async (progress, total) =>
            {
                _logger.Log(LogLevel.Information, "[OnExecuteTextToImage] - Progress: {0}/{1}, Connection: {2}", progress, total, Context.ConnectionId);
                await Clients.Caller.SendAsync("OnProgress", new ProgressResult(progress, total));
            };
        }


        private string CreateOutputImageFile(string url)
        {
            string webRootPath = _webHostEnvironment.WebRootPath;
            string physicalPath = Path.Combine(webRootPath, url.TrimStart('/').Replace('/', '\\'));
            return physicalPath;
        }

        private string CreateOutputImageUrl(string folder, string imageName)
        {
            return $"/images/results/{folder}/{imageName}";
        }
    }

    public record ProgressResult(int Progress, int Total);
    public record DiffusionResult(string OutputImage, string OutputImageUrl);
}
