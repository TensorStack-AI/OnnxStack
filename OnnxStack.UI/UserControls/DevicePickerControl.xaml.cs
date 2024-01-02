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
            DependencyProperty.Register("ExecutionProvider", typeof(ExecutionProvider), typeof(DevicePickerControl), new PropertyMetadata((c, e) =>
            {
                if (c is DevicePickerControl control)
                    control.OnSettingsChanged();
            }));

        public static readonly DependencyProperty DeviceIdProperty =
            DependencyProperty.Register("DeviceId", typeof(int), typeof(DevicePickerControl), new PropertyMetadata((c, e) =>
            {
                if (c is DevicePickerControl control)
                    control.OnSettingsChanged();
            }));

        public OnnxStackUIConfig UISettings
        {
            get { return (OnnxStackUIConfig)GetValue(UISettingsProperty); }
            set { SetValue(UISettingsProperty, value); }
        }

        /// <summary>
        /// Gets or sets the ExecutionProvider.
        /// </summary>
        public ExecutionProvider ExecutionProvider
        {
            get { return (ExecutionProvider)GetValue(ExecutionProviderProperty); }
            set { SetValue(ExecutionProviderProperty, value); }
        }

        /// <summary>
        /// Gets or sets the DeviceId.
        /// </summary>
        public int DeviceId
        {
            get { return (int)GetValue(DeviceIdProperty); }
            set { SetValue(DeviceIdProperty, value); }
        }


        public IReadOnlyList<DeviceInfo> Devices => _deviceService.Devices;



        public DeviceInfo SelectedDevice
        {
            get { return _selectedDevice; }
            set { _selectedDevice = value; OnSelectedDeviceChanged(); }
        }

        private void OnSettingsChanged()
        {
            SelectedDevice = ExecutionProvider == ExecutionProvider.Cpu
               ? Devices.FirstOrDefault()
               : Devices.FirstOrDefault(x => x.DeviceId == DeviceId);
        }

        private void OnSelectedDeviceChanged()
        {
            if (_selectedDevice.Name == "CPU")
            {
                DeviceId = 0;
                ExecutionProvider = ExecutionProvider.Cpu;
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
