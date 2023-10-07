namespace OnnxStack.WebUI.Models
{
    public record TextToImageResult(string ImageName, string ImageUrl, ImageBlueprint Blueprint, string BlueprintName, string BlueprintUrl, int Elapsed);
}
