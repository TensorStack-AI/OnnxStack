using System;
using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace OnnxStack.UI.Converters
{
    [ValueConversion(typeof(bool), typeof(Visibility))]
    public class InverseBooleanToHiddenConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            var boolValue = (bool)value;
            return !boolValue ? Visibility.Visible : Visibility.Hidden;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return null;
        }
    }
}
