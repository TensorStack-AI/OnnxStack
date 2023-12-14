using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Windows.Data;

namespace OnnxStack.UI.Converters
{
    public class ComboBoxAllItemConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            IEnumerable container = value as IEnumerable;
            if (container != null)
            {
                IEnumerable<object> genericContainer = container.OfType<object>();
                IEnumerable<object> emptyItem = new object[] { "All" };
                return emptyItem.Concat(genericContainer);
            }

            return value;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return value is string str && str == "All" ? null : value;
        }
    }
}
