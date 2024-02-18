using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Numerics.Tensors;
namespace OnnxStack.StableDiffusion.Helpers
{

    /// <summary>
    /// TODO: Optimization, some functions in here are tensor copy, but not all need to be
    /// probably some good mem/cpu gains here if a set of mutate and non-mutate functions were created
    /// </summary>
    public static class TensorHelper
    {
        private static readonly int vectorSize = Vector<float>.Count;

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
        /// <returns></returns>
        public static DenseTensor<float> DivideTensorByFloat(this DenseTensor<float> tensor, float value) => tensor.MultiplyTensorByFloat(1 / value);


        /// <summary>
        /// Multiplies the tensor by float.
        /// </summary>
        /// <param name="tensor">The data.</param>
        /// <param name="value">The value.</param>
        /// <returns></returns>
        public static DenseTensor<float> MultiplyTensorByFloat(this DenseTensor<float> tensor, float value)
        {
            var result = new DenseTensor<float>(tensor.Dimensions);
            TensorPrimitives.Multiply(tensor.Buffer.Span, value, result.Buffer.Span);
            return result;
        }


        /// <summary>
        /// Subtracts the float from each element.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="value">The value.</param>
        /// <returns></returns>
        public static DenseTensor<float> SubtractFloat(this DenseTensor<float> tensor, float value)
        {
            var result = new DenseTensor<float>(tensor.Dimensions);
            TensorPrimitives.Subtract(tensor.Buffer.Span, value, result.Buffer.Span);
            return result;
        }


        /// <summary>
        /// Adds the tensors.
        /// </summary>
        /// <param name="tensor">The sample.</param>
        /// <param name="sumTensor">The sum tensor.</param>
        /// <returns></returns>
        public static DenseTensor<float> AddTensors(this DenseTensor<float> tensor, DenseTensor<float> sumTensor)
        {
            var result = new DenseTensor<float>(tensor.Dimensions);
            TensorPrimitives.Add(tensor.Buffer.Span, sumTensor.Buffer.Span, result.Buffer.Span);
            return result;
        }


        /// <summary>
        /// Sums the tensors.
        /// </summary>
        /// <param name="tensors">The tensor array.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        public static DenseTensor<float> SumTensors(this DenseTensor<float>[] tensors, ReadOnlySpan<int> dimensions)
        {
            var result = new DenseTensor<float>(dimensions);
            for (int m = 0; m < tensors.Length; m++)
            {
                TensorPrimitives.Add(result.Buffer.Span, tensors[m].Buffer.Span, result.Buffer.Span);
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
            var result = new DenseTensor<float>(tensor.Dimensions);
            TensorPrimitives.Subtract(tensor.Buffer.Span, subTensor.Buffer.Span, result.Buffer.Span);
            return result;
        }

        /// <summary>
        /// Divides the tensor by float, mutates the original
        /// </summary>
        /// <param name="tensor">The tensor to mutate.</param>
        /// <param name="value">The value to divide by.</param>
        /// <returns></returns>
        public static DenseTensor<float> DivideBy(this DenseTensor<float> tensor, float value)
        {
            value = 1 / value;
            TensorPrimitives.Multiply(tensor.Buffer.Span, value, tensor.Buffer.Span);
            return tensor;
        }


        /// <summary>
        /// Multiples the tensor by float, mutates the original
        /// </summary>
        /// <param name="tensor">The tensor to mutate.</param>
        /// <param name="value">The value to multiply by.</param>
        /// <returns></returns>
        public static DenseTensor<float> MultiplyBy(this DenseTensor<float> tensor, float value) => DivideBy(tensor, 1 / value);


        /// <summary>
        /// Computes the absolute values of the Tensor
        /// </summary>
        /// <param name="tensor">The tensor to mutate.</param>
        /// <returns></returns>
        public static DenseTensor<float> Abs(this DenseTensor<float> tensor)
        {
            TensorPrimitives.Abs(tensor.Buffer.Span, tensor.Buffer.Span);
            return tensor;
        }


        /// <summary>
        /// Multiplies the specified tensor.
        /// </summary>
        /// <param name="tensor1">The tensor to mutate.</param>
        /// <param name="mulTensor">The tensor to multiply by.</param>
        /// <returns></returns>
        public static DenseTensor<float> Multiply(this DenseTensor<float> tensor, DenseTensor<float> mulTensor)
        {
            TensorPrimitives.Multiply(tensor.Buffer.Span, mulTensor.Buffer.Span, tensor.Buffer.Span);
            return tensor;
        }


        /// <summary>
        /// Divides the specified tensor.
        /// </summary>
        /// <param name="tensor">The tensor to mutate.</param>
        /// <param name="divTensor">The tensor to divide by.</param>
        /// <returns></returns>
        public static DenseTensor<float> Divide(this DenseTensor<float> tensor, DenseTensor<float> divTensor)
        {
            TensorPrimitives.Divide(tensor.Buffer.Span, divTensor.Buffer.Span, tensor.Buffer.Span);
            return tensor;
        }


        /// <summary>
        /// Adds the tensors, mutates the original
        /// </summary>
        /// <param name="tensor">The tensor to mutate.</param>
        /// <param name="addTensor">The tensor values to add to tensor.</param>
        /// <returns></returns>
        public static DenseTensor<float> Add(this DenseTensor<float> tensor, DenseTensor<float> addTensor)
        {
            TensorPrimitives.Add(tensor.Buffer.Span, addTensor.Buffer.Span, tensor.Buffer.Span);
            return tensor;
        }


        /// <summary>
        /// Subtracts the tensors, mutates the original
        /// </summary>
        /// <param name="tensor">The tensor to mutate.</param>
        /// <param name="subTensor">The tensor to subtract from tensor.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        public static DenseTensor<float> Subtract(this DenseTensor<float> tensor, DenseTensor<float> subTensor)
        {
            TensorPrimitives.Subtract(tensor.Buffer.Span, subTensor.Buffer.Span, tensor.Buffer.Span);
            return tensor;
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
            Vector<float> min = new Vector<float>(minValue);
            Vector<float> max = new Vector<float>(maxValue);
            var clipTensor = new DenseTensor<float>(tensor.Dimensions);
            for (int i = 0; i < tensor.Length / vectorSize; i++)
            {
                Vector.Min(min,
                    Vector.Max(max,
                    new Vector<float>(tensor.Buffer.Span.Slice(i * vectorSize, vectorSize))))
                    .CopyTo(clipTensor.Buffer.Span.Slice(i * vectorSize, vectorSize));
            }
            for (int i = (int)(tensor.Length - tensor.Length % vectorSize); i < tensor.Length; i++)
            {
                clipTensor.Buffer.Span[i] = Math.Clamp(tensor.Buffer.Span[i], minValue, maxValue);
            }
            return clipTensor;

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
                var u1 = random.NextSingle(); // Uniform(0,1) random number
                var u2 = random.NextSingle(); // Uniform(0,1) random number
                var radius = MathF.Sqrt(-2.0f * MathF.Log(u1)); // Radius of polar coordinates
                var theta = 2.0f * MathF.PI * u2; // Angle of polar coordinates
                var standardNormalRand = radius * MathF.Cos(theta); // Standard normal random number
                latents.SetValue(i, standardNormalRand * initNoiseSigma);
            }
            return latents;
        }

    }
}
