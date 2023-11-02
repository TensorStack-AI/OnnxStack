using OnnxStack.StableDiffusion.Enums;
using System;
using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace OnnxStack.UI.Converters
{
    public class PipelineVisibilityConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is DiffuserPipelineType pipeLineValue)
            {
                Enum.TryParse<DiffuserPipelineType>(parameter.ToString(), true, out var parameterEnum);
                return pipeLineValue == parameterEnum ? Visibility.Visible : Visibility.Collapsed;
            }
            return Visibility.Visible;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
