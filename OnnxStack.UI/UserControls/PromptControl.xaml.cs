using OnnxStack.UI.Commands;
using OnnxStack.UI.Models;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace OnnxStack.UI.UserControls
{
    /// <summary>
    /// Interaction logic for Parameters.xaml
    /// </summary>
    public partial class PromptControl : UserControl, INotifyPropertyChanged
    {

        /// <summary>Initializes a new instance of the <see cref="PromptControl" /> class.</summary>
        public PromptControl()
        {
            ResetParametersCommand = new RelayCommand(ResetParameters);
            InitializeComponent();
        }

        /// <summary>Gets the reset parameters command.</summary>
        /// <value>The reset parameters command.</value>
        public ICommand ResetParametersCommand { get; }


        /// <summary>
        /// Gets or sets the PromptOptions.
        /// </summary>
        public PromptOptionsModel PromptOptions
        {
            get { return (PromptOptionsModel)GetValue(PromptOptionsProperty); }
            set { SetValue(PromptOptionsProperty, value); }
        }


        /// <summary>
        /// The PromptOptions property
        /// </summary>
        public static readonly DependencyProperty PromptOptionsProperty =
            DependencyProperty.Register("PromptOptions", typeof(PromptOptionsModel), typeof(PromptControl));


        /// <summary>
        /// Resets the parameters.
        /// </summary>
        private void ResetParameters()
        {
            PromptOptions = new PromptOptionsModel();
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
