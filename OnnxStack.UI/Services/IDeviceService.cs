using System.Collections.Generic;
using OnnxStack.UI.Models;

namespace OnnxStack.UI.Services
{
    public interface IDeviceService
    {
        IReadOnlyList<DeviceInfo> Devices { get; }
    }
}