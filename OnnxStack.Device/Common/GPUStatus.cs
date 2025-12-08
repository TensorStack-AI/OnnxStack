namespace OnnxStack.Device.Common
{
    public record struct GPUStatus
    {
        public GPUStatus(GPUDevice device)
        {
            Id = device.Id;
            Name = device.Name;
            MemoryTotal = device.MemoryTotal;
            SharedMemoryTotal = device.SharedMemoryTotal;
        }

        public int Id { get; set; }
        public string Name { get; set; }

        public float MemoryTotal { get; set; }
        public float MemoryUsage { get; set; }

        public float SharedMemoryTotal { get; set; }
        public float SharedMemoryUsage { get; set; }

        public float ProcessMemoryTotal { get; set; }
        public float ProcessSharedMemoryUsage { get; set; }

        public int UsageCompute { get; set; }
        public int UsageCompute1 { get; set; }
        public int UsageGraphics { get; set; }
        public int UsageGraphics1 { get; set; }
    }
}
