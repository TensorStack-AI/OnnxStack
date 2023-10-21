using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace OnnxStack.UI.UserControls
{
    /// <summary>
    /// Interaction logic for FontAwesome.xaml
    /// </summary>
    public partial class FontAwesome : UserControl
    {
        public FontAwesome()
        {
            InitializeComponent();
        }

        public static readonly DependencyProperty SizeProperty =
            DependencyProperty.Register("Size", typeof(int), typeof(FontAwesome), new PropertyMetadata(16));

        public static readonly DependencyProperty IconProperty =
            DependencyProperty.Register("Icon", typeof(string), typeof(FontAwesome), new PropertyMetadata("\uf004"));

        public static readonly DependencyProperty ColorProperty =
            DependencyProperty.Register("Color", typeof(Brush), typeof(FontAwesome), new PropertyMetadata(Brushes.Black));

        public static readonly DependencyProperty IconStyleProperty =
            DependencyProperty.Register("IconStyle", typeof(FontAwesomeIconStyle), typeof(FontAwesome), new PropertyMetadata(FontAwesomeIconStyle.Regular));


        /// <summary>
        /// Gets or sets the icon.
        /// </summary>
        public string Icon
        {
            get { return (string)GetValue(IconProperty); }
            set { SetValue(IconProperty, value); }
        }


        /// <summary>
        /// Gets or sets the size.
        /// </summary>
        public int Size
        {
            get { return (int)GetValue(SizeProperty); }
            set { SetValue(SizeProperty, value); }
        }


        /// <summary>
        /// Gets or sets the color.
        /// </summary>
        public Brush Color
        {
            get { return (Brush)GetValue(ColorProperty); }
            set { SetValue(ColorProperty, value); }
        }


        /// <summary>
        /// Gets or sets the icon style.
        /// </summary>
        public FontAwesomeIconStyle IconStyle
        {
            get { return (FontAwesomeIconStyle)GetValue(IconStyleProperty); }
            set { SetValue(IconStyleProperty, IconStyle); }
        }
    }

    public enum FontAwesomeIconStyle
    {
        Regular,
        Light,
        Solid,
        Brands,
        Duotone
    }
}
