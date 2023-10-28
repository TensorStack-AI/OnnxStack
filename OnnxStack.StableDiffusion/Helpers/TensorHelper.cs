using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Linq;

namespace OnnxStack.StableDiffusion.Helpers
{

    /// <summary>
    /// TODO: Optimization, all functions in here are tensor copy, but not all need to be
    /// probably some good mem/cpu gains here if a set of mutate and non-mutate functions were created
    /// </summary>
    public static class TensorHelper
    {
        /// <summary>
        /// Creates a new tensor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="data">The data.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        public static DenseTensor<T> CreateTensor<T>(T[] data, ReadOnlySpan<int> dimensions)
        {
            return new DenseTensor<T>(data, dimensions);
        }


        /// <summary>
        /// Divides the tensor by float.
        /// </summary>
        /// <param name="tensor">The data.</param>
        /// <param name="value">The value.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        public static DenseTensor<float> DivideTensorByFloat(this DenseTensor<float> tensor, float value, ReadOnlySpan<int> dimensions)
        {
            var divTensor = new DenseTensor<float>(dimensions);
            for (int i = 0; i < tensor.Length; i++)
            {
                divTensor.SetValue(i, tensor.GetValue(i) / value);
            }
            return divTensor;
        }


        /// <summary>
        /// Divides the tensor by float.
        /// </summary>
        /// <param name="tensor">The data.</param>
        /// <param name="value">The value.</param>
        /// <returns></returns>
        public static DenseTensor<float> DivideTensorByFloat(this DenseTensor<float> tensor, float value)
        {
            var divTensor = new DenseTensor<float>(tensor.Dimensions);
            for (int i = 0; i < tensor.Length; i++)
            {
                divTensor.SetValue(i, tensor.GetValue(i) / value);
            }
            return divTensor;
        }


        /// <summary>
        /// Multiples the tensor by float.
        /// </summary>
        /// <param name="tensor">The data.</param>
        /// <param name="value">The value.</param>
        /// <returns></returns>
        public static DenseTensor<float> MultipleTensorByFloat(this DenseTensor<float> tensor, float value)
        {
            var mullTensor = new DenseTensor<float>(tensor.Dimensions);
            for (int i = 0; i < tensor.Length; i++)
            {
                mullTensor.SetValue(i, tensor.GetValue(i) * value);
            }
            return mullTensor;
        }


        /// <summary>
        /// Subtracts the float from each element.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="value">The value.</param>
        /// <returns></returns>
        public static DenseTensor<float> SubtractFloat(this DenseTensor<float> tensor, float value)
        {
            var subTensor = new DenseTensor<float>(tensor.Dimensions);
            for (int i = 0; i < tensor.Length; i++)
            {
                subTensor.SetValue(i, tensor.GetValue(i) - value);
            }
            return subTensor;
        }

        /// <summary>
        /// Adds the tensors.
        /// </summary>
        /// <param name="tensor">The sample.</param>
        /// <param name="sumTensor">The sum tensor.</param>
        /// <returns></returns>
        public static DenseTensor<float> AddTensors(this DenseTensor<float> tensor, DenseTensor<float> sumTensor)
        {
            var addTensor = new DenseTensor<float>(tensor.Dimensions);
            for (var i = 0; i < tensor.Length; i++)
            {
                addTensor.SetValue(i, tensor.GetValue(i) + sumTensor.GetValue(i));
            }
            return addTensor;
        }





        /// <summary>
        /// Sums the tensors.
        /// </summary>
        /// <param name="tensors">The tensor array.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        public static DenseTensor<float> SumTensors(this DenseTensor<float>[] tensors, ReadOnlySpan<int> dimensions)
        {
            var sumTensor = new DenseTensor<float>(dimensions);
            for (int m = 0; m < tensors.Length; m++)
            {
                var tensorToSum = tensors[m];
                for (var i = 0; i < tensorToSum.Length; i++)
                {
                    sumTensor.SetValue(i, sumTensor.GetValue(i) + tensorToSum.GetValue(i));
                }
            }
            return sumTensor;
        }


        /// <summary>
        /// Duplicates the specified tensor.
        /// </summary>
        /// <param name="tensor">The data.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        public static DenseTensor<float> Duplicate(this DenseTensor<float> tensor, ReadOnlySpan<int> dimensions)
        {
            var dupTensor = tensor.Concat(tensor).ToArray();
            return CreateTensor(dupTensor, dimensions);
        }


        /// <summary>
        /// Subtracts the tensors.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="subTensor">The sub tensor.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        public static DenseTensor<float> SubtractTensors(this DenseTensor<float> tensor, DenseTensor<float> subTensor, ReadOnlySpan<int> dimensions)
        {
            var result = new DenseTensor<float>(dimensions);
            for (var i = 0; i < tensor.Length; i++)
            {
                result.SetValue(i, tensor.GetValue(i) - subTensor.GetValue(i));
            }
            return result;
        }


        /// <summary>
        /// Subtracts the tensors.
        /// </summary>
        /// <param name="tensor">The sample.</param>
        /// <param name="subTensor">The sub tensor.</param>
        /// <returns></returns>
        public static DenseTensor<float> SubtractTensors(this DenseTensor<float> tensor, DenseTensor<float> subTensor)
        {
            return tensor.SubtractTensors(subTensor, tensor.Dimensions);
        }


        /// <summary>
        /// Reorders the tensor.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <returns></returns>
        public static DenseTensor<float> ReorderTensor(this DenseTensor<float> tensor, ReadOnlySpan<int> dimensions)
        {
            //reorder from batch channel height width to batch height width channel
            var inputImagesTensor = new DenseTensor<float>(dimensions);
            for (int y = 0; y < tensor.Dimensions[2]; y++)
            {
                for (int x = 0; x < tensor.Dimensions[3]; x++)
                {
                    inputImagesTensor[0, y, x, 0] = tensor[0, 0, y, x];
                    inputImagesTensor[0, y, x, 1] = tensor[0, 1, y, x];
                    inputImagesTensor[0, y, x, 2] = tensor[0, 2, y, x];
                }
            }
            return inputImagesTensor;
        }



        /// <summary>
        /// Clips the specified Tensor valuse to the specified minimum/maximum.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="minValue">The minimum value.</param>
        /// <param name="maxValue">The maximum value.</param>
        /// <returns></returns>
        public static DenseTensor<float> Clip(this DenseTensor<float> tensor, float minValue, float maxValue)
        {
            var clipTensor = new DenseTensor<float>(tensor.Dimensions);
            for (int i = 0; i < tensor.Length; i++)
            {
                clipTensor.SetValue(i, Math.Clamp(tensor.GetValue(i), minValue, maxValue));
            }
            return clipTensor;
        }


        /// <summary>
        /// Computes the absolute values of the Tensor
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns></returns>
        public static DenseTensor<float> Abs(this DenseTensor<float> tensor)
        {
            var absTensor = new DenseTensor<float>(tensor.Dimensions);
            for (int i = 0; i < tensor.Length; i++)
            {
                absTensor.SetValue(i, Math.Abs(tensor.GetValue(i)));
            }
            return absTensor;
        }


        /// <summary>
        /// Multiplies the specified tensor.
        /// </summary>
        /// <param name="tensor1">The tensor1.</param>
        /// <param name="tensor2">The tensor2.</param>
        /// <returns></returns>
        public static DenseTensor<float> Multiply(this DenseTensor<float> tensor1, DenseTensor<float> tensor2)
        {
            var result = new DenseTensor<float>(tensor1.Dimensions);
            for (int batch = 0; batch < tensor1.Dimensions[0]; batch++)
            {
                for (int channel = 0; channel < tensor1.Dimensions[1]; channel++)
                {
                    for (int height = 0; height < tensor1.Dimensions[2]; height++)
                    {
                        for (int width = 0; width < tensor1.Dimensions[3]; width++)
                        {
                            var value1 = tensor1[batch, channel, height, width];
                            var value2 = tensor2[batch, channel, height, width];
                            result[batch, channel, height, width] = value1 * value2;
                        }
                    }
                }
            }
            return result;
        }


        /// <summary>
        /// Divides the specified tensor.
        /// </summary>
        /// <param name="tensor1">The tensor1.</param>
        /// <param name="tensor2">The tensor2.</param>
        /// <returns></returns>
        public static DenseTensor<float> Divide(this DenseTensor<float> tensor1, DenseTensor<float> tensor2)
        {
            var result = new DenseTensor<float>(tensor1.Dimensions);
            for (int batch = 0; batch < tensor1.Dimensions[0]; batch++)
            {
                for (int channel = 0; channel < tensor1.Dimensions[1]; channel++)
                {
                    for (int height = 0; height < tensor1.Dimensions[2]; height++)
                    {
                        for (int width = 0; width < tensor1.Dimensions[3]; width++)
                        {
                            var value1 = tensor1[batch, channel, height, width];
                            var value2 = tensor2[batch, channel, height, width];
                            result[batch, channel, height, width] = value1 / value2;
                        }
                    }
                }
            }
            return result;
        }


        /// <summary>
        /// Generate a random Tensor from a normal distribution with mean 0 and variance 1
        /// </summary>
        /// <param name="random">The random.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <param name="initNoiseSigma">The initialize noise sigma.</param>
        /// <returns></returns>
        public static DenseTensor<float> GetRandomTensor(Random random, ReadOnlySpan<int> dimensions, float initNoiseSigma = 1f)
        {
            var latents = new DenseTensor<float>(dimensions);
            for (int i = 0; i < latents.Length; i++)
            {
                // Generate a random number from a normal distribution with mean 0 and variance 1
                var u1 = random.NextDouble(); // Uniform(0,1) random number
                var u2 = random.NextDouble(); // Uniform(0,1) random number
                var radius = Math.Sqrt(-2.0 * Math.Log(u1)); // Radius of polar coordinates
                var theta = 2.0 * Math.PI * u2; // Angle of polar coordinates
                var standardNormalRand = radius * Math.Cos(theta); // Standard normal random number
                latents.SetValue(i, (float)standardNormalRand * initNoiseSigma);
            }
            return latents;
        }
    }
}
