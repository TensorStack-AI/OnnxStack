using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Runtime.CompilerServices;

namespace OnnxStack.UI.Models
{
    public class PromptOptionsModel : INotifyPropertyChanged
    {
        private string _prompt;
        private string _negativePrompt;
        private bool _hasChanged;

        [Required]
        [StringLength(512, MinimumLength = 1)]
        public string Prompt
        {
            get { return _negativePrompt; }
            set { _negativePrompt = value; NotifyPropertyChanged(); }
        }

        [StringLength(512)]
        public string NegativePrompt
        {
            get { return _prompt; }
            set { _prompt = value; NotifyPropertyChanged(); }
        }

        public bool HasChanged
        {
            get { return _hasChanged; }
            set { _hasChanged = value; NotifyPropertyChanged(); }
        }


        #region INotifyPropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;
        public void NotifyPropertyChanged([CallerMemberName] string property = "")
        {
            if (!property.Equals(nameof(HasChanged)) && !HasChanged)
                HasChanged = true;

            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(property));
        }
        #endregion

    }
}
