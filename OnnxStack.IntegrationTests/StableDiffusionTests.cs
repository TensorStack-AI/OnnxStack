using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using Xunit.Abstractions;

namespace OnnxStack.IntegrationTests;

public class StableDiffusionTests
{
    private readonly ITestOutputHelper _testOutputHelper;
    private readonly IStableDiffusionService _stableDiffusion;

    public StableDiffusionTests(ITestOutputHelper testOutputHelper)
    {
        _testOutputHelper = testOutputHelper;
        
        var services = new ServiceCollection();
        services.AddLogging();
        services.AddOnnxStack();
        services.AddOnnxStackStableDiffusion();
        var provider = services.BuildServiceProvider();
        _stableDiffusion = provider.GetRequiredService<IStableDiffusionService>();
    }
    
    [Fact]
    public async Task GivenStableDiffusion15_WhenLoadModel_ThenModelIsLoaded()
    {
        //arrange
        var model = _stableDiffusion.Models.Single(m => m.Name == "StableDiffusion 1.5");
        
        //act
        var isModelLoaded = await _stableDiffusion.LoadModel(model);

        //assert
        isModelLoaded.Should().BeTrue();
    }
    
    [Fact]
    public async Task GivenTextToImage_WhenInference_ThenImageGenerated()
    {
        //arrange
        var model = _stableDiffusion.Models.Single(m => m.Name == "StableDiffusion 1.5");
        await _stableDiffusion.LoadModel(model);
        
        var prompt = new PromptOptions
        {
            Prompt = "an astronaut riding a horse in space",
            NegativePrompt = "blurry,ugly,cartoon",
            BatchCount = 1,
            SchedulerType = SchedulerType.EulerAncestral,
            DiffuserType = DiffuserType.TextToImage
        };

        var scheduler = new SchedulerOptions
        {
            Width = 512,
            Height = 512,
            InferenceSteps = 10,
            GuidanceScale = 7.0f,
            Seed = -1
        };

        var steps = 0;
        
        //act
        var image = await _stableDiffusion.GenerateAsImageAsync(model, prompt, scheduler, (currentStep, totalSteps) =>
        {
            _testOutputHelper.WriteLine($"Step {currentStep}/{totalSteps}");
            steps++;
        });

        //assert
        steps.Should().Be(10);
        image.Should().NotBeNull();
        image.Size.IsEmpty.Should().BeFalse();
        image.Width.Should().Be(512);
        image.Height.Should().Be(512);
    }
}