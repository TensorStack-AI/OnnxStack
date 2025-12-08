using System.ComponentModel;

namespace OnnxStack.StableDiffusion.Enums
{
    public enum DiffuserType
    {
        [Description("Text To Image")]
        TextToImage = 0,

        [Description("Image To Image")]
        ImageToImage = 1,

        [Description("Image Inpaint")]
        ImageInpaint = 2,

        [Description("Image Inpaint Legacy")]
        ImageInpaintLegacy = 3,

        [Description("Text To Video")]
        TextToVideo = 4,

        [Description("Image To Video")]
        ImageToVideo = 5,

        [Description("Video To Video")]
        VideoToVideo = 6,

        [Description("ControlNet")]
        ControlNet = 100,

        [Description("ControlNet Image")]
        ControlNetImage = 101,

        [Description("ControlNet Video")]
        ControlNetVideo = 102
    }
}
