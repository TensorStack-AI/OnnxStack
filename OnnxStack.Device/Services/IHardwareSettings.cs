namespace OnnxStack.Device.Services
{
    public interface IHardwareSettings
    {
        public int ProcessId { get; set; }
        public bool UseLegacyDeviceDetection { get; set; }
    }
}
