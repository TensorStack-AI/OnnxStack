using System.Security.Cryptography;
using FluentAssertions;
using FluentAssertions.Execution;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using SixLabors.ImageSharp;
using Xunit.Abstractions;

namespace OnnxStack.IntegrationTests;

/// <summary>
/// These tests just run on CPU execution provider for now, but could switch it to CUDA and run on GPU
/// if the necessary work is done to setup the docker container to allow GPU passthrough to the container.
/// See https://blog.roboflow.com/use-the-gpu-in-docker/ for an example of how to do this.
///
/// Can then also setup a self-hosted runner in Github Actions to run the tests on your own GPU as part of the CI/CD pipeline.
/// Maybe something like https://www.youtube.com/watch?v=rVq-SCNyxVc
/// </summary>
public class StableDiffusionTests
{
    private readonly IStableDiffusionService _stableDiffusion;
    private readonly ILogger<StableDiffusionTests> _logger;

    public StableDiffusionTests(ITestOutputHelper testOutputHelper)
    {
        var services = new ServiceCollection();
        services.AddLogging(builder => builder.AddConsole());
        services.AddLogging(builder => builder.AddXunit(testOutputHelper));
        services.AddOnnxStack();
        services.AddOnnxStackStableDiffusion();
        var provider = services.BuildServiceProvider();
        _stableDiffusion = provider.GetRequiredService<IStableDiffusionService>();
        _logger = provider.GetRequiredService<ILogger<StableDiffusionTests>>();
    }
    
    [Fact]
    public async Task GivenStableDiffusion15_WhenLoadModel_ThenModelIsLoaded()
    {
        //arrange
        var model = _stableDiffusion.Models.Single(m => m.Name == "StableDiffusion 1.5");
        
        //act
        _logger.LogInformation("Attempting to load model {0}", model.Name);
        var isModelLoaded = await _stableDiffusion.LoadModel(model);

        //assert
        isModelLoaded.Should().BeTrue();
    }
    
    [Fact]
    public async Task GivenTextToImage_WhenInference_ThenImageGenerated()
    {
        //arrange
        var model = _stableDiffusion.Models.Single(m => m.Name == "StableDiffusion 1.5");
        _logger.LogInformation("Attempting to load model {0}", model.Name);
        await _stableDiffusion.LoadModel(model);
        
        var prompt = new PromptOptions
        {
            Prompt = "an astronaut riding a horse in space",
            NegativePrompt = "blurry,ugly,cartoon",
            BatchCount = 1,
            DiffuserType = DiffuserType.TextToImage
        };

        var scheduler = new SchedulerOptions
        {
            Width = 512,
            Height = 512,
            SchedulerType = SchedulerType.EulerAncestral,
            InferenceSteps = 10,
            GuidanceScale = 7.0f,
            Seed = 1
        };

        var steps = 0;
        
        //act
        var image = await _stableDiffusion.GenerateAsImageAsync(model, prompt, scheduler, (currentStep, totalSteps) =>
        {
            _logger.LogInformation($"Step {currentStep}/{totalSteps}");
            steps++;
        });

        var imagesDirectory = Path.Combine(Directory.GetCurrentDirectory(), "images");
        if (!Directory.Exists(imagesDirectory))
        {
            _logger.LogInformation($"Creating directory {imagesDirectory}");
            Directory.CreateDirectory(imagesDirectory);
        }
        else
        {
            _logger.LogInformation($"Directory {imagesDirectory} already exists");
        }

        var fileName = $"{imagesDirectory}/{nameof(GivenTextToImage_WhenInference_ThenImageGenerated)}-{DateTime.Now:yyyyMMddHHmmss}.png";
        _logger.LogInformation($"Saving generated image to {fileName}");
        await image.SaveAsPngAsync(fileName);

        //assert
        using (new AssertionScope())
        {
            steps.Should().Be(10);
            image.Should().NotBeNull();
            image.Size.IsEmpty.Should().BeFalse();
            image.Width.Should().Be(512);
            image.Height.Should().Be(512);
        
            File.Exists(fileName).Should().BeTrue();
            var md5 = MD5.Create();
            var hash = md5.ComputeHash(File.ReadAllBytes(fileName));
            var hashString = string.Join("", hash.Select(b => b.ToString("X2")));
            _logger.LogInformation($"MD5 Hash of generated image: {hashString}");
            
            hashString.Should().Be("E518D0E4F67CBD5E93513574D30F3FD7");
        }
    }
}