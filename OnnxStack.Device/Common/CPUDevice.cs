namespace OnnxStack.Device.Common
{
    public record CPUDevice
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public float MemoryTotal { get; set; }
    }
}
