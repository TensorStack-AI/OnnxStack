using OnnxStack.UI.Commands;
using Microsoft.Win32;
using System.Globalization;
using System.IO;
using System.Windows;
using System.Windows.Controls;

namespace OnnxStack.UI.UserControls
{
    /// <summary>
    /// Interaction logic for FilePickerTextBox.xaml
    /// </summary>
    public partial class FilePickerTextBox : UserControl
    {
        public FilePickerTextBox()
        {
            ClearFileCommand = new RelayCommand(ClearFile, CanExecuteClearFile);
            OpenFileDialogCommand = new RelayCommand(OpenPicker, CanExecuteOpenPicker);
            InitializeComponent();
        }

        public static readonly DependencyProperty FileNameProperty =
           DependencyProperty.Register("FileName", typeof(string), typeof(FilePickerTextBox));

        public static readonly DependencyProperty FilterProperty =
            DependencyProperty.Register("Filter", typeof(string), typeof(FilePickerTextBox));

        public static readonly DependencyProperty TitleProperty =
            DependencyProperty.Register("Title", typeof(string), typeof(FilePickerTextBox));

        public static readonly DependencyProperty InitialDirectoryProperty =
            DependencyProperty.Register("InitialDirectory", typeof(string), typeof(FilePickerTextBox));

        public static readonly DependencyProperty DefaultExtProperty =
            DependencyProperty.Register("DefaultExt", typeof(string), typeof(FilePickerTextBox));

        public static readonly DependencyProperty IsRequiredProperty =
            DependencyProperty.Register("IsRequired", typeof(bool), typeof(FilePickerTextBox));

        public static readonly DependencyProperty IsFolderPickerProperty =
            DependencyProperty.Register("IsFolderPicker", typeof(bool), typeof(FilePickerTextBox));


        /// <summary>
        /// Gets or sets the clear file command.
        /// </summary>
        public RelayCommand ClearFileCommand { get; set; }

        /// <summary>
        /// Gets or sets the open file dialog command.
        /// </summary>
        public RelayCommand OpenFileDialogCommand { get; set; }


        /// <summary>
        /// Gets or sets a string containing the file name selected in the file dialog box.
        /// </summary>
        public string FileName
        {
            get { return (string)GetValue(FileNameProperty); }
            set { SetValue(FileNameProperty, value); }
        }


        /// <summary>
        /// Gets or sets the file dialog box title.
        /// </summary>
        public string Title
        {
            get { return (string)GetValue(TitleProperty); }
            set { SetValue(TitleProperty, value); }
        }


        /// <summary>
        /// Gets or sets the initial directory displayed by the file dialog box.
        /// </summary>
        public string InitialDirectory
        {
            get { return (string)GetValue(InitialDirectoryProperty); }
            set { SetValue(InitialDirectoryProperty, value); }
        }

        /// <summary>
        /// Gets or sets the current file name filter string, which determines the choices that appear in the "Save as file type" or "Files of type" box in the dialog box.
        /// </summary>
        /// <example>"Text files (*.txt)|*.txt|All files (*.*)|*.*";</example>
        public string Filter
        {
            get { return (string)GetValue(FilterProperty); }
            set { SetValue(FilterProperty, value); }
        }


        /// <summary>
        /// Gets or sets the default ext.
        /// </summary>
        /// <value>
        /// The default file name extension. The returned string does not include the period. The default value is an empty string ("").
        /// </value>
        public string DefaultExt
        {
            get { return (string)GetValue(DefaultExtProperty); }
            set { SetValue(DefaultExtProperty, value); }
        }


        /// <summary>
        /// Gets or sets a value indicating whether this file is required.
        /// </summary>
        /// <value>
        ///   <c>true</c> if this instance is required; otherwise, <c>false</c>.
        /// </value>
        public bool IsRequired
        {
            get { return (bool)GetValue(IsRequiredProperty); }
            set { SetValue(IsRequiredProperty, value); }
        }


        /// <summary>
        /// Gets or sets a value indicating whether this instance is folder picker.
        /// </summary>
        /// <value>
        ///   <c>true</c> if this instance is folder picker; otherwise, <c>false</c>.
        /// </value>
        public bool IsFolderPicker
        {
            get { return (bool)GetValue(IsFolderPickerProperty); }
            set { SetValue(IsFolderPickerProperty, value); }
        }


        /// <summary>
        /// Opens the picker.
        /// </summary>
        private void OpenPicker()
        {
            if (IsFolderPicker)
            {
                OpenFolderPicker();
                return;
            }

            OpenFilePicker();
        }


        /// <summary>
        /// Determines whether this instance can execute OpenPicker.
        /// </summary>
        /// <returns>
        ///   <c>true</c> if this instance can execute OpenPicker; otherwise, <c>false</c>.
        /// </returns>
        private bool CanExecuteOpenPicker()
        {
            return true;
        }


        /// <summary>
        /// Opens the file picker.
        /// </summary>
        private void OpenFilePicker()
        {
            var openFileDialog = new OpenFileDialog
            {
                Title = Title,
                Filter = Filter,
                CheckFileExists = true,
                InitialDirectory = InitialDirectory,
                DefaultExt = DefaultExt
            };
            var dialogResult = openFileDialog.ShowDialog();
            if (dialogResult == true)
                FileName = openFileDialog.FileName;
        }


        /// <summary>
        /// Opens the folder picker.
        /// </summary>
        private void OpenFolderPicker()
        {
            var folderBrowserDialog = new System.Windows.Forms.FolderBrowserDialog
            {
                Description = Title,
                InitialDirectory = InitialDirectory,
                UseDescriptionForTitle = true
            };
            var dialogResult = folderBrowserDialog.ShowDialog();
            if (dialogResult == System.Windows.Forms.DialogResult.OK)
                FileName = folderBrowserDialog.SelectedPath;
        }


        /// <summary>
        /// Clears the file.
        /// </summary>
        private void ClearFile()
        {
            FileName = string.Empty;
        }


        /// <summary>
        /// Determines whether this instance can execute ClearFile.
        /// </summary>
        /// <returns>
        ///   <c>true</c> if this instance can execute ClearFile; otherwise, <c>false</c>.
        /// </returns>
        private bool CanExecuteClearFile()
        {
            return !IsRequired;
        }
    }


    /// <summary>
    /// ValidationRule to check if a file exists if its required
    /// </summary>
    /// <seealso cref="System.Windows.Controls.ValidationRule" />
    public class FileExistsValidationRule : ValidationRule
    {
        public bool IsFolder { get; set; }
        public bool IsRequired { get; set; }

        public override ValidationResult Validate(object value, CultureInfo cultureInfo)
        {
            var filename = value?.ToString() ?? string.Empty;
            if (!IsRequired && string.IsNullOrEmpty(filename))
                return ValidationResult.ValidResult;

            if (!IsFolder && !File.Exists(filename))
                return new ValidationResult(false, $"File does not exist");

            if (IsFolder && !Directory.Exists(filename))
                return new ValidationResult(false, $"Directory does not exist");

            return ValidationResult.ValidResult;
        }
    }
}
