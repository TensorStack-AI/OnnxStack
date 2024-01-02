namespace OnnxStack.UI.Models
{
    public class DeviceInfo
    {
        public DeviceInfo(string name, int deviceId, int vram)
        {
            Name = name;
            DeviceId = deviceId;
            VRAM = vram;
        }

        public string Name { get; set; }
        public int DeviceId { get; set; }
        public int VRAM { get; set; }
    }
}
