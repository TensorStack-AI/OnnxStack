using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Models
{
    public record DiffusionProgress(int ProgressValue, int ProgressMax, DenseTensor<float> ProgressTensor)
    {
        public int SubProgressMax { get; set; }
        public int SubProgressValue { get; set; }
    }
  
}
