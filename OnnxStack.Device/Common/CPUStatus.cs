namespace OnnxStack.Device.Common
{
    public record struct CPUStatus
    {
        public CPUStatus(CPUDevice device)
        {
            Id = device.Id;
            Name = device.Name;
            MemoryTotal = device.MemoryTotal;
        }

        public int Id { get; }
        public string Name { get; }

        public float MemoryTotal { get; }




        public int Usage { get; set; }
        public int MemoryUsage { get; set; }
        public float MemoryAvailable { get; set; }
    }
}
