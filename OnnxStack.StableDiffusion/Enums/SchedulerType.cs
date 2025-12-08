using System.ComponentModel.DataAnnotations;
using System.Xml.Linq;

namespace OnnxStack.StableDiffusion.Enums
{
    public enum SchedulerType
    {
        [Display(Name = "LMS")]
        LMS = 0,

        [Display(Name = "Euler")]
        Euler = 1,

        [Display(Name = "Euler-Ancestral")]
        EulerAncestral = 2,

        [Display(Name = "DDPM")]
        DDPM = 3,

        [Display(Name = "DDIM")]
        DDIM = 4,

        [Display(Name = "KDPM2")]
        KDPM2 = 5,

        [Display(Name = "KDPM2-Ancestral")]
        KDPM2Ancestral = 6,

        [Display(Name = "DDPM-Wuerstchen")]
        DDPMWuerstchen = 10,

        [Display(Name = "LCM")]
        LCM = 20,

        [Display(Name = "FlowMatch-EulerDiscrete")]
        FlowMatchEulerDiscrete = 22,

        [Display(Name = "FlowMatch-EulerDynamic")]
        FlowMatchEulerDynamic = 23,

        [Display(Name = "Locomotion")]
        Locomotion = 30,




        // Deprecated 
        FlashFlowMatchEulerDiscrete = 1000
    }
}
