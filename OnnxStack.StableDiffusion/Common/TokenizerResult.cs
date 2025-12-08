namespace OnnxStack.StableDiffusion.Common
{
    public record TokenizerResult
    {
        public TokenizerResult(long[] inputIds, long[] attentionMask)
        {
            InputIds = inputIds;
            AttentionMask = attentionMask;
        }

        public long[] InputIds { get; set; }
        public long[] AttentionMask { get; set; }
    }
}
