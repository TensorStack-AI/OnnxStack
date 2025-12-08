using OnnxStack.Device.Common;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Management;
using System.Threading;

namespace OnnxStack.Device.Services
{
    public sealed class HardwareService : IHardwareService
    {
        private readonly int _refreshRate = 500;
        private readonly CPUDevice _cpuDevice;
        private readonly NPUDevice _npuDevice;
        private readonly GPUDevice[] _gpuDevices;
        private readonly ManagementObjectSearcher _objectSearcherDriver;
        private readonly ManagementObjectSearcher _objectSearcherProcessor;
        private readonly ManagementObjectSearcher _objectSearcherGPUEngine;
        private readonly ManagementObjectSearcher _objectSearcherGPUMemory;
        private readonly ManagementObjectSearcher _objectSearcherGPUProcessMemory;
        private readonly ManagementObjectSearcher _objectSearcherProcessorPercent;
        private readonly ManualResetEvent _updateThreadResetEvent;
        private readonly CancellationTokenSource _cancellationTokenSource;
        private Thread _cpuUpdateThread;
        private Thread _gpuUpdateThread;
        private CPUStatus _cpuStatus;
        private NPUStatus _npuStatus;
        private GPUStatus[] _gpuStatus;
        private AdapterInfo[] _adapters;
        private DeviceInfo[] _deviceInfo;

        public HardwareService(IHardwareSettings hardwareSettings)
        {
            _cancellationTokenSource = new CancellationTokenSource();
            _objectSearcherDriver = new ManagementObjectSearcher("root\\CIMV2", "SELECT DeviceName, DriverVersion FROM Win32_PnPSignedDriver");
            _objectSearcherProcessor = new ManagementObjectSearcher("root\\CIMV2", "SELECT Name FROM Win32_Processor");
            _objectSearcherGPUEngine = new ManagementObjectSearcher("root\\CIMV2", $"SELECT Name, UtilizationPercentage FROM Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine WHERE Name LIKE 'pid_{hardwareSettings.ProcessId}%'");
            _objectSearcherGPUMemory = new ManagementObjectSearcher("root\\CIMV2", "SELECT Name, SharedUsage, DedicatedUsage, TotalCommitted FROM Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory");
            _objectSearcherGPUProcessMemory = new ManagementObjectSearcher("root\\CIMV2", $"SELECT Name, SharedUsage, DedicatedUsage, TotalCommitted FROM Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory WHERE Name LIKE 'pid_{hardwareSettings.ProcessId}%'");
            _objectSearcherProcessorPercent = new ManagementObjectSearcher("root\\CIMV2", "SELECT PercentProcessorTime FROM Win32_PerfFormattedData_PerfOS_Processor WHERE Name='_Total'");
            _adapters = hardwareSettings.UseLegacyDeviceDetection
                ? DeviceInterop.GetAdaptersLegacy()
                : DeviceInterop.GetAdapters();
            _deviceInfo = GetDeviceInfo();
            _cpuDevice = CreateDeviceCPU();
            _gpuDevices = CreateDeviceGPU();
            _npuDevice = CreateDeviceNPU();
            _updateThreadResetEvent = new ManualResetEvent(true);
            CreateUpdateThreadCPU(_refreshRate, _cancellationTokenSource.Token);
            CreateUpdateThreadGPU(_refreshRate, _cancellationTokenSource.Token);
        }


        public CPUDevice CPUDevice => _cpuDevice;
        public CPUStatus CPUStatus => _cpuStatus;

        public NPUDevice NPUDevice => _npuDevice;
        public NPUStatus NPUStatus => _npuStatus;

        public GPUDevice[] GPUDevices => _gpuDevices ?? [];
        public GPUStatus[] GPUStatus => _gpuStatus ?? [];

        public AdapterInfo[] Adapters => _adapters ?? [];


        private CPUDevice CreateDeviceCPU()
        {
            try
            {
                string cpuName = "Unknown CPU";
                using (var results = _objectSearcherProcessor.Get())
                {
                    foreach (var result in results)
                    {
                        using (result)
                        {
                            cpuName = result["Name"].ToString().Trim();
                        }
                    }
                }

                var deviceInfo = _deviceInfo.FirstOrDefault(d => d.Name == cpuName);
                var memStatus = DeviceInterop.GetMemoryStatus();
                return new CPUDevice
                {
                    Name = cpuName,
                    MemoryTotal = memStatus.TotalPhysicalMemory / 1024f / 1024f
                };
            }
            catch (Exception)
            {
                return default;
            }
        }


        private CPUStatus CreateStatusCPU()
        {
            try
            {
                ulong processorUsage = 0;
                using (var results = _objectSearcherProcessorPercent.Get())
                {
                    foreach (var result in results)
                    {
                        using (result)
                        {
                            processorUsage = (ulong)result["PercentProcessorTime"];
                        }
                    }
                }

                var memStatus = DeviceInterop.GetMemoryStatus();
                return new CPUStatus(_cpuDevice)
                {
                    Usage = (int)processorUsage,
                    MemoryUsage = (int)memStatus.MemoryLoad,
                    MemoryAvailable = memStatus.AvailablePhysicalMemory / 1024f / 1024f
                };
            }
            catch (Exception)
            {
                return default;
            }
        }


        private GPUDevice[] CreateDeviceGPU()
        {
            try
            {
                var devices = new List<GPUDevice>();
                foreach (var device in _adapters.Where(x => x.Type == AdapterType.GPU))
                {
                    var deviceInfo = _deviceInfo.FirstOrDefault(d => d.Name == device.Description);
                    devices.Add(new GPUDevice(device, deviceInfo));
                }
                return devices.ToArray();
            }
            catch (Exception)
            {
                return [];
            }
        }


        private GPUStatus[] CreateStatusGPU(Dictionary<string, GPUUtilization[]> gpuUtilization, Dictionary<string, GPUMemory> gpuMemoryUsage, Dictionary<string, GPUMemory> gpuProcessUsage)
        {
            var results = new GPUStatus[_gpuDevices.Length];
            for (int i = 0; i < _gpuDevices.Length; i++)
            {
                try
                {
                    var device = _gpuDevices[i];
                    var result = new GPUStatus(device);

                    if (gpuMemoryUsage.TryGetValue(device.AdapterId, out var memoryUsage))
                    {
                        result.MemoryUsage = memoryUsage.DedicatedUsage / 1024f / 1024f;
                        result.SharedMemoryUsage = memoryUsage.SharedUsage / 1024f / 1024f;
                    }

                    if (gpuProcessUsage.TryGetValue(device.AdapterId, out var processUsage))
                    {
                        result.ProcessMemoryTotal = processUsage.DedicatedUsage / 1024f / 1024f;
                        result.ProcessSharedMemoryUsage = processUsage.SharedUsage / 1024f / 1024f;
                    }

                    if (gpuUtilization.TryGetValue(device.AdapterId, out var deviceUtilization))
                    {
                        var utilization = deviceUtilization
                            .GroupBy(x => x.Instance)
                            .ToDictionary(u => u.Key, y => y.Sum(x => (int)x.Utilization));
                        utilization.TryGetValue("3D", out var engineGraphics);
                        utilization.TryGetValue("Graphics1", out var engineGraphics1);
                        utilization.TryGetValue("Compute", out var engineCompute);
                        utilization.TryGetValue("Compute1", out var engineCompute1);
                        result.UsageCompute = engineCompute;
                        result.UsageCompute1 = engineCompute1;
                        result.UsageGraphics = engineGraphics;
                        result.UsageGraphics1 = engineGraphics1;
                    }

                    results[i] = result;
                }
                catch (Exception)
                {

                }
            }
            return results;
        }


        private NPUDevice CreateDeviceNPU()
        {
            try
            {
                var devices = new List<NPUDevice>();
                foreach (var device in _adapters.Where(x => x.IsHardware && x.Type == AdapterType.NPU))
                {
                    var deviceInfo = _deviceInfo.FirstOrDefault(d => d.Name == device.Description);
                    devices.Add(new NPUDevice(device, deviceInfo));
                }
                return devices.FirstOrDefault();
            }
            catch (Exception)
            {
                return default;
            }
        }


        private NPUStatus CreateStatusNPU(Dictionary<string, GPUUtilization[]> gpuUtilization, Dictionary<string, GPUMemory> gpuMemoryUsage, Dictionary<string, GPUMemory> gpuProcessUsage)
        {
            try
            {
                if (_npuDevice == default)
                    return default;

                var result = new NPUStatus(_npuDevice);
                if (gpuMemoryUsage.TryGetValue(_npuDevice.AdapterId, out var memoryUsage))
                {
                    result.MemoryUsage = memoryUsage.SharedUsage / 1024f / 1024f;
                }

                if (gpuProcessUsage.TryGetValue(_npuDevice.AdapterId, out var processUsage))
                {
                    result.ProcessMemoryTotal = processUsage.DedicatedUsage / 1024f / 1024f;
                    result.ProcessSharedMemoryUsage = processUsage.SharedUsage / 1024f / 1024f;
                }

                if (gpuUtilization.TryGetValue(_npuDevice.AdapterId, out var deviceUtilization))
                {
                    var utilization = deviceUtilization
                        .GroupBy(x => x.Instance)
                        .ToDictionary(u => u.Key, y => y.Sum(x => (int)x.Utilization));
                    utilization.TryGetValue("Compute", out var engineCompute);
                    result.Usage = engineCompute > 100 ? engineCompute - 100 : engineCompute;
                }
                return result;
            }
            catch (Exception)
            {
                return default;
            }
        }


        private Dictionary<string, GPUMemory> GetMemoryUsageGPU()
        {
            var gpuMemory = new Dictionary<string, GPUMemory>();
            try
            {
                using (var results = _objectSearcherGPUMemory.Get())
                {
                    foreach (var result in results)
                    {
                        using (result)
                        {
                            var adapterId = result["Name"]?.ToString();
                            var sharedUsage = (ulong)result.Properties["SharedUsage"].Value;
                            var dedicatedUsage = (ulong)result.Properties["DedicatedUsage"].Value;
                            var totalCommitted = (ulong)result.Properties["TotalCommitted"].Value;
                            gpuMemory.Add(adapterId, new GPUMemory(adapterId, sharedUsage, dedicatedUsage, totalCommitted));
                        }
                    }
                }
            }
            catch (Exception)
            {

            }
            return gpuMemory;
        }



        private Dictionary<string, GPUMemory> GetProcessMemoryUsageGPU()
        {
            var gpuMemory = new Dictionary<string, GPUMemory>();
            try
            {
                using (var results = _objectSearcherGPUProcessMemory.Get())
                {
                    foreach (var result in results)
                    {
                        using (result)
                        {
                            var parts = result.Properties["Name"].Value.ToString().Split('_');
                            var adapterId = string.Join('_', parts.Skip(2));
                            var sharedUsage = (ulong)result.Properties["SharedUsage"].Value;
                            var dedicatedUsage = (ulong)result.Properties["DedicatedUsage"].Value;
                            var totalCommitted = (ulong)result.Properties["TotalCommitted"].Value;
                            gpuMemory.Add(adapterId, new GPUMemory(adapterId, sharedUsage, dedicatedUsage, totalCommitted));
                        }
                    }
                }
            }
            catch (Exception)
            {

            }
            return gpuMemory;
        }



        private Dictionary<string, GPUUtilization[]> GetUtilizationGPU()
        {
            var gpuUtilization = new List<GPUUtilization>();
            try
            {
                using (var results = _objectSearcherGPUEngine.Get())
                {
                    foreach (var result in results)
                    {
                        using (result)
                        {
                            var percentage = (ulong)result.Properties["UtilizationPercentage"].Value;
                            if (percentage == 0)
                                continue;

                            string instanceName = result["Name"]?.ToString();
                            if (instanceName.Contains("engtype_3D") || instanceName.Contains("engtype_Graphics") || instanceName.Contains("engtype_Compute"))
                            {
                                var parts = result.Properties["Name"].Value.ToString().Split('_');
                                int.TryParse(parts[1], out int pid);
                                var instance = string.Concat(parts.Skip(10).Take(3));
                                var adapterId = string.Join('_', parts.Skip(2).Take(5));
                                gpuUtilization.Add(new GPUUtilization(adapterId, instance, percentage, pid));
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {

            }

            return gpuUtilization
                .GroupBy(x => x.AdapterId)
                .ToDictionary(k => k.Key, v => v.ToArray());
        }


        private DeviceInfo[] GetDeviceInfo()
        {
            try
            {
                var versions = new List<DeviceInfo>();
                var devices = _adapters.Select(x => x.Description).ToArray();

                using (var results = _objectSearcherDriver.Get())
                {
                    foreach (var result in results)
                    {
                        var deviceName = result["DeviceName"]?.ToString() ?? string.Empty;
                        if (!devices.Contains(deviceName))
                            continue;

                        var driverVersion = result["DriverVersion"]?.ToString() ?? string.Empty;
                        versions.Add(new DeviceInfo(deviceName.Trim(), driverVersion));
                    }
                }

                return versions.ToArray();
            }
            catch (Exception)
            {
                return [];
            }
        }


        private void CreateUpdateThreadCPU(int refreshRate, CancellationToken cancellationToken)
        {
            _cpuUpdateThread = new Thread(() =>
            {
                while (true)
                {
                    try
                    {
                        _updateThreadResetEvent.WaitOne();
                        cancellationToken.ThrowIfCancellationRequested();

                        var timestamp = Stopwatch.GetTimestamp();
                        _cpuStatus = CreateStatusCPU();
                        var eplased = Stopwatch.GetElapsedTime(timestamp).TotalMilliseconds;
                        Thread.Sleep(refreshRate);
                    }
                    catch (OperationCanceledException) { break; }
                    catch (ThreadInterruptedException) { break; }
                }
            });
            _cpuUpdateThread.IsBackground = true;
            _cpuUpdateThread.Start();
        }


        private void CreateUpdateThreadGPU(int refreshRate, CancellationToken cancellationToken)
        {
            _gpuUpdateThread = new Thread(() =>
            {
                while (true)
                {
                    try
                    {
                        _updateThreadResetEvent.WaitOne();
                        cancellationToken.ThrowIfCancellationRequested();

                        var timestamp = Stopwatch.GetTimestamp();
                        var gpuProcessMemoryUsage = GetProcessMemoryUsageGPU();
                        var gpuMemoryUsage = GetMemoryUsageGPU();
                        var gpuUtilization = GetUtilizationGPU();
                        _gpuStatus = CreateStatusGPU(gpuUtilization, gpuMemoryUsage, gpuProcessMemoryUsage);
                        _npuStatus = CreateStatusNPU(gpuUtilization, gpuMemoryUsage, gpuProcessMemoryUsage);
                        var eplased = Stopwatch.GetElapsedTime(timestamp).TotalMilliseconds;
                        Thread.Sleep(refreshRate);
                    }
                    catch (OperationCanceledException) { break; }
                    catch (ThreadInterruptedException) { break; }
                }
            });
            _gpuUpdateThread.IsBackground = true;
            _gpuUpdateThread.Start();
        }


        /// <summary>
        /// Pauses this instance.
        /// </summary>
        public void Pause()
        {
            _updateThreadResetEvent.Reset();
        }


        /// <summary>
        /// Resumes this instance.
        /// </summary>
        public void Resume()
        {
            _updateThreadResetEvent.Set();
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            _cancellationTokenSource.Cancel();
            _updateThreadResetEvent.Set();
            _cpuUpdateThread.Join();
            _gpuUpdateThread.Join();
            _updateThreadResetEvent.Dispose();
            _cancellationTokenSource.Dispose();
        }

        private record struct GPUUtilization(string AdapterId, string Instance, ulong Utilization, int ProcessId);
        private record struct GPUMemory(string AdapterId, ulong SharedUsage, ulong DedicatedUsage, ulong TotalCommitted);
    }
}
