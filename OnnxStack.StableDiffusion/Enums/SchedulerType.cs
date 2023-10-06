using System.ComponentModel.DataAnnotations;

namespace OnnxStack.StableDiffusion.Enums
{
    public enum SchedulerType
    {
        [Display(Name = "LMS")]
        LMS = 0,

        [Display(Name = "Euler Ancestral")]
        EulerAncestral = 1,

        [Display(Name = "DDPM")]
        DDPM = 3
    }
}
