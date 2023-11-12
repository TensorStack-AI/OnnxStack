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
    }

    public enum BatchOptionType
    {
        Seed = 0
    }
}
