using OnnxStack.Device.Common;
using System;

namespace OnnxStack.Device.Services
{
    public interface IHardwareService : IDisposable
    {
        CPUDevice CPUDevice { get; }
        CPUStatus CPUStatus { get; }

        NPUDevice NPUDevice { get; }
        NPUStatus NPUStatus { get; }

        GPUDevice[] GPUDevices { get; }
        GPUStatus[] GPUStatus { get; }

        AdapterInfo[] Adapters { get; }

        void Pause();
        void Resume();
    }
}
