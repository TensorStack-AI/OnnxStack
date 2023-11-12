using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Config
{
    public class BatchOptions
    {
        public BatchOptionType BatchType { get; set; }
        public int Count { get; set; }

        public float ValueTo { get; set; }
        public float ValueFrom { get; set; }
        public float Increment { get; set; } = 1f;
    }

    public enum BatchOptionType
    {
        Seed = 0,
        Step = 1,
        Guidance = 2
    }
}
