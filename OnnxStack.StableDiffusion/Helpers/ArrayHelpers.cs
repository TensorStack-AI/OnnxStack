using System;
using System.Linq;

namespace OnnxStack.StableDiffusion.Helpers
{
    internal class ArrayHelpers
    {
        public static float[] Linspace(float start, float end, int partitions, bool round = false)
        {
            var result = Enumerable.Range(0, partitions)
                .Select(idx => idx != partitions ? start + (end - start) / (partitions - 1) * idx : end);
            return !round
                ? result.ToArray()
                : result.Select(x => MathF.Round(x)).ToArray();
        }


        public static float[] Range(int start, int end)
        {
            return Enumerable.Range(start, end)
                .Select(x => (float)x)
                .ToArray();
        }

        public static float[] Log(float[] array)
        {
            return array
                .Select(x => MathF.Log(x))
                .ToArray();
        }
    }
}
