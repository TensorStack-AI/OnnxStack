namespace OnnxStack.Device.Common
{
    public record struct NPUStatus
    {
        public NPUStatus(NPUDevice device)
        {
            Id = device.Id;
            Name = device.Name;
            MemoryTotal = device.MemoryTotal;
        }

        public int Id { get; set; }
        public string Name { get; set; }

        public int Usage { get; set; }
        public float MemoryTotal { get; set; }
        public float MemoryUsage { get; set; }

        public float ProcessMemoryTotal { get; set; }
        public float ProcessSharedMemoryUsage { get; set; }
    }
}
