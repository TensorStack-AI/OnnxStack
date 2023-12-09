namespace OnnxStack.Core.Config
{
    public enum OnnxModelType
    {
        Unet = 0,
        Tokenizer = 10,
        Tokenizer2 = 11,
        TextEncoder = 20,
        TextEncoder2 = 21,
        VaeEncoder = 30,
        VaeDecoder = 40,

        Upscaler = 100
    }
}
