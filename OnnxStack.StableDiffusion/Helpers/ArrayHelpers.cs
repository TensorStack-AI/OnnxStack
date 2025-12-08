using System;
using System.Linq;

namespace OnnxStack.StableDiffusion.Helpers
{
    internal class ArrayHelpers
    {
        public static float[] Linspace(float start, float end, int partitions, bool round = false)
        {
            var result = Enumerable.Range(0, partitions)
                .Select(idx => idx != partitions ? start + (end - start) / (partitions - 1) * idx : end)
                .Select(x => float.IsNaN(x) ? 0.0f : x);
            return !round
                ? result.ToArray()
                : result.Select(x => MathF.Round(x)).ToArray();
        }


        public static float[] Range(int start, int end, bool reverse = false)
        {
            var range = Enumerable.Range(start, end)
                .Select(x => (float)x)
                .ToArray();
            if (reverse)
                Array.Reverse(range);
            return range;
        }

        public static float[] Log(float[] array)
        {
            return array
                .Select(x => MathF.Log(x))
                .ToArray();
        }


        public static int BinarySearchDescending(float[] array, float value)
        {
            int low = 0;
            int high = array.Length - 1;

            while (low <= high)
            {
                int mid = (low + high) / 2;

                if (MathF.Abs(array[mid] - value) < 1e-6f)
                    return mid;

                if (array[mid] < value)
                    high = mid - 1;
                else
                    low = mid + 1;
            }

            return ~low;
        }
    }
}
