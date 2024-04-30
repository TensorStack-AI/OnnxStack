using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Numerics;
using OnnxStack.Core.Model;
using System.Threading.Tasks;

namespace OnnxStack.Core
{
    public static class TensorExtension
    {
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

        private static readonly int vectorSize = Vector<float>.Count;

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
        public static DenseTensor<float> NextTensor(this Random random, ReadOnlySpan<int> dimensions, float initNoiseSigma = 1f)
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

        /// <summary>
        /// Repeats the specified Tensor along the 0 axis.
        /// </summary>
        /// <param name="tensor1">The tensor1.</param>
        /// <param name="count">The count.</param>
        /// <param name="axis">The axis.</param>
        /// <returns></returns>
        /// <exception cref="System.NotImplementedException">Only axis 0 is supported</exception>
        public static DenseTensor<float> Repeat(this DenseTensor<float> tensor1, int count, int axis = 0)
        {
            if (axis != 0)
                throw new NotImplementedException("Only axis 0 is supported");

            var dimensions = tensor1.Dimensions.ToArray();
            dimensions[0] *= count;

            var length = (int)tensor1.Length;
            var totalLength = length * count;
            var buffer = new float[totalLength].AsMemory();
            for (int i = 0; i < count; i++)
            {
                tensor1.Buffer.CopyTo(buffer[(i * length)..]);
            }
            return new DenseTensor<float>(buffer, dimensions);
        }


        /// <summary>
        /// Normalize the data using Min-Max scaling to ensure all values are in the range [0, 1].
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        public static void NormalizeMinMax(this DenseTensor<float> tensor)
        {
            tensor.Buffer.Span.NormalizeZeroToOne();
        }


        /// <summary>
        /// Splits the tensor across the batch dimension.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns></returns>
        public static IEnumerable<DenseTensor<float>> SplitBatch(this DenseTensor<float> tensor)
        {
            var count = tensor.Dimensions[0];
            var dimensions = tensor.Dimensions.ToArray();
            dimensions[0] = 1;

            var newLength = (int)tensor.Length / count;
            for (int i = 0; i < count; i++)
            {
                var start = i * newLength;
                yield return new DenseTensor<float>(tensor.Buffer.Slice(start, newLength), dimensions);
            }
        }


        /// <summary>
        /// Joins the tensors across the 0 axis.
        /// </summary>
        /// <param name="tensors">The tensors.</param>
        /// <param name="axis">The axis.</param>
        /// <returns></returns>
        /// <exception cref="System.NotImplementedException">Only axis 0 is supported</exception>
        public static DenseTensor<float> Join(this IList<DenseTensor<float>> tensors, int axis = 0)
        {
            if (axis != 0)
                throw new NotImplementedException("Only axis 0 is supported");

            var tensor = tensors.First();
            var dimensions = tensor.Dimensions.ToArray();
            dimensions[0] *= tensors.Count;

            var newLength = (int)tensor.Length;
            var buffer = new float[newLength * tensors.Count].AsMemory();
            for (int i = 0; i < tensors.Count(); i++)
            {
                var start = i * newLength;
                tensors[i].Buffer.CopyTo(buffer[start..]);
            }
            return new DenseTensor<float>(buffer, dimensions);
        }


        /// <summary>
        /// Concatenates the specified tensors along the specified axis.
        /// </summary>
        /// <param name="tensor1">The tensor1.</param>
        /// <param name="tensor2">The tensor2.</param>
        /// <param name="axis">The axis.</param>
        /// <returns></returns>
        /// <exception cref="System.NotImplementedException">Only axis 0,1,2 is supported</exception>
        public static DenseTensor<float> Concatenate(this DenseTensor<float> tensor1, DenseTensor<float> tensor2, int axis = 0)
        {
            if (tensor1 == null)
                return tensor2.ToDenseTensor();

            return axis switch
            {
                0 => ConcatenateAxis0(tensor1, tensor2),
                1 => ConcatenateAxis1(tensor1, tensor2),
                2 => ConcatenateAxis2(tensor1, tensor2),
                _ => throw new NotImplementedException("Only axis 0, 1, 2 is supported")
            };
        }


        private static DenseTensor<float> ConcatenateAxis0(this DenseTensor<float> tensor1, DenseTensor<float> tensor2)
        {
            var dimensions = tensor1.Dimensions.ToArray();
            dimensions[0] += tensor2.Dimensions[0];

            var buffer = new DenseTensor<float>(dimensions);
            tensor1.Buffer.CopyTo(buffer.Buffer[..(int)tensor1.Length]);
            tensor2.Buffer.CopyTo(buffer.Buffer[(int)tensor1.Length..]);
            return buffer;
        }


        private static DenseTensor<float> ConcatenateAxis1(DenseTensor<float> tensor1, DenseTensor<float> tensor2)
        {
            var dimensions = tensor1.Dimensions.ToArray();
            dimensions[1] += tensor2.Dimensions[1];
            var concatenatedTensor = new DenseTensor<float>(dimensions);

            // Copy data from the first tensor
            for (int i = 0; i < dimensions[0]; i++)
                for (int j = 0; j < tensor1.Dimensions[1]; j++)
                    concatenatedTensor[i, j] = tensor1[i, j];

            // Copy data from the second tensor
            for (int i = 0; i < dimensions[0]; i++)
                for (int j = 0; j < tensor2.Dimensions[1]; j++)
                    concatenatedTensor[i, j + tensor1.Dimensions[1]] = tensor2[i, j];

            return concatenatedTensor;
        }


        private static DenseTensor<float> ConcatenateAxis2(DenseTensor<float> tensor1, DenseTensor<float> tensor2)
        {
            var dimensions = tensor1.Dimensions.ToArray();
            dimensions[2] += tensor2.Dimensions[2];
            var concatenatedTensor = new DenseTensor<float>(dimensions);

            // Copy data from the first tensor
            for (int i = 0; i < dimensions[0]; i++)
                for (int j = 0; j < dimensions[1]; j++)
                    for (int k = 0; k < tensor1.Dimensions[2]; k++)
                        concatenatedTensor[i, j, k] = tensor1[i, j, k];

            // Copy data from the second tensor
            for (int i = 0; i < dimensions[0]; i++)
                for (int j = 0; j < dimensions[1]; j++)
                    for (int k = 0; k < tensor2.Dimensions[2]; k++)
                        concatenatedTensor[i, j, k + tensor1.Dimensions[2]] = tensor2[i, j, k];

            return concatenatedTensor;
        }


        /// <summary>
        /// Normalizes the tensor values from range -1 to 1 to 0 to 1.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        public static void NormalizeOneOneToZeroOne(this DenseTensor<float> imageTensor)
        {
            Parallel.For(0, (int)imageTensor.Length, (i) => imageTensor.SetValue(i, imageTensor.GetValue(i) / 2f + 0.5f));
        }


        /// <summary>
        /// Normalizes the tensor values from range 0 to 1 to -1 to 1.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        public static void NormalizeZeroOneToOneOne(this DenseTensor<float> imageTensor)
        {
            Parallel.For(0, (int)imageTensor.Length, (i) => imageTensor.SetValue(i, 2f * imageTensor.GetValue(i) - 1f));
        }
    }
}
