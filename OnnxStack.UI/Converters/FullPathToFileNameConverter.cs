using System;
using System.Globalization;
using System.IO;
using System.Windows.Data;

namespace OnnxStack.UI.Converters
{
    [ValueConversion(typeof(string), typeof(string))]
    public class FullPathToFileNameConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is not string fullPath)
                return value;

            return Path.GetFileName(fullPath);
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
