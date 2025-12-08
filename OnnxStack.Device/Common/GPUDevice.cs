namespace OnnxStack.Device.Common
{
    public record GPUDevice
    {
        public GPUDevice(AdapterInfo adapter, DeviceInfo deviceInfo)
        {
            Id = (int)adapter.Id;
            AdapterInfo = adapter;
            Name = adapter.Description;
            DriverVersion = deviceInfo.DriverVersion;
            MemoryTotal = adapter.DedicatedVideoMemory / 1024f / 1024f;
            SharedMemoryTotal = adapter.SharedSystemMemory / 1024f / 1024f;
            AdapterId = $"luid_0x{adapter.AdapterLuid.HighPart:X8}_0x{adapter.AdapterLuid.LowPart:X8}_phys_0";
        }

        public int Id { get; }
        public string Name { get; }
        public string AdapterId { get; }
        public string DriverVersion { get; }
        public float MemoryTotal { get; }
        public float SharedMemoryTotal { get; }
        public AdapterInfo AdapterInfo { get; }
    }
}
