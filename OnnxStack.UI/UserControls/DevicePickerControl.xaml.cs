using OnnxStack.Core.Config;
using OnnxStack.UI.Models;
using OnnxStack.UI.Services;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Controls;

namespace OnnxStack.UI.UserControls
{
    /// <summary>
    /// Interaction logic for DevicePickerControl.xaml
    /// </summary>
    public partial class DevicePickerControl : UserControl, INotifyPropertyChanged
    {
        private readonly IDeviceService _deviceService;
        private DeviceInfo _selectedDevice;

        /// <summary>
        /// Initializes a new instance of the <see cref="DevicePickerControl"/> class.
        /// </summary>
        public DevicePickerControl()
        {
            if (!DesignerProperties.GetIsInDesignMode(this))
                _deviceService = App.GetService<IDeviceService>();

            InitializeComponent();
        }

        public static readonly DependencyProperty UISettingsProperty =
             DependencyProperty.Register("UISettings", typeof(OnnxStackUIConfig), typeof(DevicePickerControl), new PropertyMetadata((c, e) =>
             {
                 if (c is DevicePickerControl control)
                     control.OnSettingsChanged();
             }));

        public static readonly DependencyProperty ExecutionProviderProperty =
            DependencyProperty.Register("ExecutionProvider", typeof(ExecutionProvider?), typeof(DevicePickerControl), new PropertyMetadata((c, e) =>
            {
                if (c is DevicePickerControl control)
                    control.OnSettingsChanged();
            }));

        public static readonly DependencyProperty DeviceIdProperty =
            DependencyProperty.Register("DeviceId", typeof(int?), typeof(DevicePickerControl), new PropertyMetadata((c, e) =>
            {
                if (c is DevicePickerControl control)
                    control.OnSettingsChanged();
            }));


        /// <summary>
        /// Gets or sets the UI settings.
        /// </summary>
        public OnnxStackUIConfig UISettings
        {
            get { return (OnnxStackUIConfig)GetValue(UISettingsProperty); }
            set { SetValue(UISettingsProperty, value); }
        }

        /// <summary>
        /// Gets or sets the ExecutionProvider.
        /// </summary>
        public ExecutionProvider? ExecutionProvider
        {
            get { return (ExecutionProvider?)GetValue(ExecutionProviderProperty); }
            set { SetValue(ExecutionProviderProperty, value); }
        }

        /// <summary>
        /// Gets or sets the DeviceId.
        /// </summary>
        public int? DeviceId
        {
            get { return (int?)GetValue(DeviceIdProperty); }
            set { SetValue(DeviceIdProperty, value); }
        }

        /// <summary>
        /// Gets the devices.
        /// </summary>
        public IReadOnlyList<DeviceInfo> Devices => _deviceService.Devices;


        /// <summary>
        /// Gets or sets the selected device.
        /// </summary>
        public DeviceInfo SelectedDevice
        {
            get { return _selectedDevice; }
            set { _selectedDevice = value; OnSelectedDeviceChanged(); }
        }


        /// <summary>
        /// Called when UISettings changed.
        /// </summary>
        private void OnSettingsChanged()
        {
            SelectedDevice = ExecutionProvider == Core.Config.ExecutionProvider.Cpu
               ? Devices.FirstOrDefault()
               : Devices.FirstOrDefault(x => x.Name != "CPU" && x.DeviceId == DeviceId);
        }


        /// <summary>
        /// Called when SelectedDevice changed.
        /// </summary>
        private void OnSelectedDeviceChanged()
        {
            if (_selectedDevice is null)
                return;

            if (_selectedDevice.Name == "CPU")
            {
                DeviceId = 0;
                ExecutionProvider = Core.Config.ExecutionProvider.Cpu;
            }
            else
            {
                DeviceId = _selectedDevice.DeviceId;
                ExecutionProvider = UISettings.SupportedExecutionProvider;
            }

            NotifyPropertyChanged(nameof(SelectedDevice));
        }

        #region INotifyPropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;

        public void NotifyPropertyChanged([CallerMemberName] string property = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(property));
        }
        #endregion
    }
}
