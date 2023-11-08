using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace OnnxStack.UI.Behaviors
{
    /// <summary>
    /// Behaviour to use Shift + Enfer to add a new line to a TextBox allowing IsDefault Commands to be fired on Enter
    /// </summary>
    public class ShiftEnterBehavior
    {

        /// <summary>
        /// The enable property
        /// </summary>
        public static readonly DependencyProperty EnableProperty = DependencyProperty.RegisterAttached("Enable", typeof(bool), typeof(ShiftEnterBehavior), new PropertyMetadata(false, OnEnableChanged));


        /// <summary>
        /// Gets the enable value.
        /// </summary>
        /// <param name="obj">The object.</param>
        public static bool GetEnable(DependencyObject obj)
        {
            return (bool)obj.GetValue(EnableProperty);
        }

        /// <summary>
        /// Sets the enable valse.
        /// </summary>
        /// <param name="obj">The object.</param>
        /// <param name="value">if set to <c>true</c> [value].</param>
        public static void SetEnable(DependencyObject obj, bool value)
        {
            obj.SetValue(EnableProperty, value);
        }


        /// <summary>
        /// Called when enable changed.
        /// </summary>
        /// <param name="obj">The object.</param>
        /// <param name="e">The <see cref="DependencyPropertyChangedEventArgs"/> instance containing the event data.</param>
        private static void OnEnableChanged(DependencyObject obj, DependencyPropertyChangedEventArgs e)
        {
            if (obj is TextBox textBox)
            {
                bool attach = (bool)e.NewValue;

                if (attach)
                {
                    textBox.PreviewKeyDown += TextBox_PreviewKeyDown;
                }
                else
                {
                    textBox.PreviewKeyDown -= TextBox_PreviewKeyDown;
                }
            }
        }


        /// <summary>
        /// Handles the PreviewKeyDown event of the TextBox control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="KeyEventArgs"/> instance containing the event data.</param>
        private static void TextBox_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter && Keyboard.Modifiers == ModifierKeys.Shift)
            {
                if (sender is TextBox textBox)
                {
                    e.Handled = true;
                    textBox.AppendText(Environment.NewLine);
                    textBox.CaretIndex = textBox.Text.Length;
                }
            }
        }
    }
}
