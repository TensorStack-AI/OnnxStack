using OnnxStack.StableDiffusion.Config;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Data;

namespace OnnxStack.UI.Converters
{
    public class DiffuserVisibilityConverter : IMultiValueConverter
    {
        public object Convert(object[] values, Type targetType, object parameter, CultureInfo culture)
        {
            if (values.Length == 2 && values[0] is List<DiffuserType> viewTypes && values[1] is List<DiffuserType> modelTypes)
            {
                return viewTypes.Any(modelTypes.Contains) ? Visibility.Visible : Visibility.Hidden;
            }

            return Visibility.Hidden; 
        }

        public object[] ConvertBack(object value, Type[] targetTypes, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
