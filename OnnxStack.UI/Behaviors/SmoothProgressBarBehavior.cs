using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Animation;

namespace OnnxStack.UI.Behaviors
{
    public class SmoothProgressBarBehavior
    {
        public static double GetSmoothValue(DependencyObject obj)
        {
            return (double)obj.GetValue(SmoothValueProperty);
        }

        public static void SetSmoothValue(DependencyObject obj, double value)
        {
            obj.SetValue(SmoothValueProperty, value);
        }

        public static readonly DependencyProperty SmoothValueProperty =
            DependencyProperty.RegisterAttached("SmoothValue", typeof(double), typeof(SmoothProgressBarBehavior), new PropertyMetadata(0.0, changing));

        private static void changing(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            var anim = new DoubleAnimation((double)e.OldValue, (double)e.NewValue, new TimeSpan(0, 0, 0, 0, 200));
            (d as ProgressBar).BeginAnimation(ProgressBar.ValueProperty, anim, HandoffBehavior.Compose);
        }
    }
}
