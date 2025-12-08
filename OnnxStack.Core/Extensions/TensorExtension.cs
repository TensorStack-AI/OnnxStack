using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Image;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Numerics.Tensors;
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
        public static DenseTensor<float> DivideTensorByFloat(this DenseTensor<float> tensor, float value)
        {
            if (value == 0)
                return tensor.CloneTensor();

            var result = new DenseTensor<float>(tensor.Dimensions);
            TensorPrimitives.Divide(tensor.Buffer.Span, value, result.Buffer.Span);
            return result;
        }

        /// <summary>
        /// Multiplies the tensor by float.
        /// </summary>
        /// <param name="tensor">The data.</param>
        /// <param name="value">The value.</param>
        /// <returns></returns>
        public static DenseTensor<float> MultiplyTensorByFloat(this DenseTensor<float> tensor, float value)
        {
            if (value == 1)
                return tensor.CloneTensor();

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
            if (value == 0)
                return tensor.CloneTensor();

            var result = new DenseTensor<float>(tensor.Dimensions);
            TensorPrimitives.Subtract(tensor.Buffer.Span, value, result.Buffer.Span);
            return result;
        }

        /// <summary>
        /// Adds the float value to the tensor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="value">The value.</param>
        /// <returns></returns>
        public static DenseTensor<float> AddFloat(this DenseTensor<float> tensor, float value)
        {
            if (value == 0)
                return tensor.CloneTensor();

            var result = new DenseTensor<float>(tensor.Dimensions);
            TensorPrimitives.Add(tensor.Buffer.Span, value, tensor.Buffer.Span);
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
            if (value == 0)
                return tensor;

            TensorPrimitives.Divide(tensor.Buffer.Span, value, tensor.Buffer.Span);
            return tensor;
        }


        /// <summary>
        /// Multiples the tensor by float, mutates the original
        /// </summary>
        /// <param name="tensor">The tensor to mutate.</param>
        /// <param name="value">The value to multiply by.</param>
        /// <returns></returns>
        public static DenseTensor<float> MultiplyBy(this DenseTensor<float> tensor, float value)
        {
            if (value == 0)
                return tensor;

            TensorPrimitives.Multiply(tensor.Buffer.Span, value, tensor.Buffer.Span);
            return tensor;
        }


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
        /// Adds the tensors, mutates the original
        /// </summary>
        /// <param name="tensor">The tensor to mutate.</param>
        /// <param name="value">The value to add to tensor.</param>
        /// <returns></returns>
        public static DenseTensor<float> Add(this DenseTensor<float> tensor, float value)
        {
            if (value == 0)
                return tensor;

            TensorPrimitives.Add(tensor.Buffer.Span, value, tensor.Buffer.Span);
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
        /// Subtracts the specified value, mutates the original
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="value">The value.</param>
        public static DenseTensor<float> Subtract(this DenseTensor<float> tensor, float value)
        {
            if (value == 0)
                return tensor;

            TensorPrimitives.Subtract(tensor.Buffer.Span, value, tensor.Buffer.Span);
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
        public static DenseTensor<T> Repeat<T>(this DenseTensor<T> tensor1, int count, int axis = 0)
        {
            if (count == 1)
                return tensor1;

            if (axis != 0)
                throw new NotImplementedException("Only axis 0 is supported");

            var dimensions = tensor1.Dimensions.ToArray();
            dimensions[0] *= count;

            var length = (int)tensor1.Length;
            var totalLength = length * count;
            var buffer = new T[totalLength].AsMemory();
            for (int i = 0; i < count; i++)
            {
                tensor1.Buffer.CopyTo(buffer[(i * length)..]);
            }
            return new DenseTensor<T>(buffer, dimensions);
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
        /// Split first tensor from batch and return
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns></returns>
        public static DenseTensor<float> FirstBatch(this DenseTensor<float> tensor)
        {
            return SplitBatch(tensor).FirstOrDefault();
        }


        /// <summary>
        /// Lasts the batch.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>DenseTensor&lt;System.Single&gt;.</returns>
        public static DenseTensor<float> LastBatch(this DenseTensor<float> tensor)
        {
            return SplitBatch(tensor).LastOrDefault();
        }


        /// <summary>
        /// Reshapes the specified dimensions.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        public static DenseTensor<float> ReshapeTensor(this DenseTensor<float> tensor, ReadOnlySpan<int> dimensions)
        {
            return tensor.Reshape(dimensions) as DenseTensor<float>;
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
                return tensor2.CloneTensor();

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

            if (tensor1.Dimensions.Length == 2)
            {
                for (int i = 0; i < tensor1.Dimensions[0]; i++)
                    for (int j = 0; j < tensor1.Dimensions[1]; j++)
                        concatenatedTensor[i, j] = tensor1[i, j];

                for (int i = 0; i < tensor1.Dimensions[0]; i++)
                    for (int j = 0; j < tensor2.Dimensions[1]; j++)
                        concatenatedTensor[i, j + tensor1.Dimensions[1]] = tensor2[i, j];
            }
            else if (tensor1.Dimensions.Length == 3)
            {
                for (int i = 0; i < tensor1.Dimensions[0]; i++)
                    for (int j = 0; j < tensor1.Dimensions[1]; j++)
                        for (int k = 0; k < tensor1.Dimensions[2]; k++)
                            concatenatedTensor[i, j, k] = tensor1[i, j, k];

                for (int i = 0; i < tensor2.Dimensions[0]; i++)
                    for (int j = 0; j < tensor2.Dimensions[1]; j++)
                        for (int k = 0; k < tensor2.Dimensions[2]; k++)
                            concatenatedTensor[i, j + tensor1.Dimensions[1], k] = tensor2[i, j, k];
            }
            else if (tensor1.Dimensions.Length == 4)
            {
                for (int i = 0; i < tensor1.Dimensions[0]; i++)
                    for (int j = 0; j < tensor1.Dimensions[1]; j++)
                        for (int k = 0; k < tensor1.Dimensions[2]; k++)
                            for (int l = 0; l < tensor1.Dimensions[3]; l++)
                                concatenatedTensor[i, j, k, l] = tensor1[i, j, k, l];

                for (int i = 0; i < tensor2.Dimensions[0]; i++)
                    for (int j = 0; j < tensor2.Dimensions[1]; j++)
                        for (int k = 0; k < tensor2.Dimensions[2]; k++)
                            for (int l = 0; l < tensor2.Dimensions[3]; l++)
                                concatenatedTensor[i, j + tensor1.Dimensions[1], k, l] = tensor2[i, j, k, l];
            }
            else
            {
                throw new ArgumentException("Length 2 or 3 currently supported");
            }

            return concatenatedTensor;
        }


        private static DenseTensor<float> ConcatenateAxis2(DenseTensor<float> tensor1, DenseTensor<float> tensor2)
        {
            var dimensions = tensor1.Dimensions.ToArray();
            dimensions[2] += tensor2.Dimensions[2];
            var concatenatedTensor = new DenseTensor<float>(dimensions);

            for (int i = 0; i < dimensions[0]; i++)
                for (int j = 0; j < dimensions[1]; j++)
                    for (int k = 0; k < tensor1.Dimensions[2]; k++)
                        concatenatedTensor[i, j, k] = tensor1[i, j, k];

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
        public static DenseTensor<float> NormalizeOneOneToZeroOne(this DenseTensor<float> imageTensor)
        {
            Parallel.For(0, (int)imageTensor.Length, (i) => imageTensor.SetValue(i, imageTensor.GetValue(i) / 2f + 0.5f));
            return imageTensor;
        }


        /// <summary>
        /// Normalizes the tensor values from range 0 to 1 to -1 to 1.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        public static DenseTensor<float> NormalizeZeroOneToOneOne(this DenseTensor<float> imageTensor)
        {
            Parallel.For(0, (int)imageTensor.Length, (i) => imageTensor.SetValue(i, 2f * imageTensor.GetValue(i) - 1f));
            return imageTensor;
        }


        /// <summary>
        /// Pads the end dimenison by the specified length.
        /// </summary>
        /// <param name="tensor1">The tensor1.</param>
        /// <param name="padLength">Length of the pad.</param>
        /// <returns></returns>
        public static DenseTensor<float> PadEnd(this DenseTensor<float> tensor1, int padLength)
        {
            var dimensions = tensor1.Dimensions.ToArray();
            dimensions[^1] += padLength;
            var concatenatedTensor = new DenseTensor<float>(dimensions);

            if (tensor1.Dimensions.Length == 2)
            {
                for (int i = 0; i < tensor1.Dimensions[0]; i++)
                    for (int j = 0; j < tensor1.Dimensions[1]; j++)
                        concatenatedTensor[i, j] = tensor1[i, j];
            }
            else if (tensor1.Dimensions.Length == 3)
            {
                for (int i = 0; i < tensor1.Dimensions[0]; i++)
                    for (int j = 0; j < tensor1.Dimensions[1]; j++)
                        for (int k = 0; k < tensor1.Dimensions[2]; k++)
                            concatenatedTensor[i, j, k] = tensor1[i, j, k];
            }
            else
            {
                throw new ArgumentException("Length 2 or 3 currently supported");
            }

            return concatenatedTensor;
        }


        /// <summary>
        /// Permutes the dimensions of a tensor according to the specified permutation order.
        /// </summary>
        /// <typeparam name="T">The type of the tensor elements.</typeparam>
        /// <param name="tensor">The input tensor to permute.</param>
        /// <param name="permutation">An array specifying the permutation order of the dimensions.</param>
        /// <returns>A new tensor with permuted dimensions.</returns>
        public static DenseTensor<T> Permute<T>(this DenseTensor<T> tensor, int[] permutation)
        {
            // Get the original shape of the tensor
            int[] originalShape = tensor.Dimensions.ToArray();

            // Create a new shape based on the permutation
            int[] newShape = permutation.Select(i => originalShape[i]).ToArray();

            // Create a new tensor with the permuted shape
            DenseTensor<T> permutedTensor = new DenseTensor<T>(newShape);

            // Calculate indices for both original and permuted tensors
            int[] originalIndex = new int[originalShape.Length];
            int[] permutedIndex = new int[newShape.Length];

            // Populate the permuted tensor with values from the original tensor
            for (int i = 0; i < tensor.Length; i++)
            {
                // Get the multi-dimensional index for the original tensor
                int remaining = i;
                for (int j = originalShape.Length - 1; j >= 0; j--)
                {
                    originalIndex[j] = remaining % originalShape[j];
                    remaining /= originalShape[j];
                }

                // Apply the permutation to get the new index
                for (int j = 0; j < newShape.Length; j++)
                {
                    permutedIndex[j] = originalIndex[permutation[j]];
                }

                // Calculate the flat index for the permuted tensor
                int permutedFlatIndex = 0;
                int multiplier = 1;
                for (int j = newShape.Length - 1; j >= 0; j--)
                {
                    permutedFlatIndex += permutedIndex[j] * multiplier;
                    multiplier *= newShape[j];
                }

                // Assign the value from the original tensor to the permuted tensor
                permutedTensor.Buffer.Span[permutedFlatIndex] = tensor.Buffer.Span[i];
            }

            return permutedTensor;
        }

        public static DenseTensor<float> SoftMax(this DenseTensor<float> tesnor)
        {
            TensorPrimitives.SoftMax(tesnor.Buffer.Span, tesnor.Buffer.Span);
            return tesnor;
        }

        public static DenseTensor<T> Ones<T>(ReadOnlySpan<int> dimensions) where T : INumber<T> => Fill(dimensions, T.One);
        public static DenseTensor<T> Zeros<T>(ReadOnlySpan<int> dimensions) where T : INumber<T> => Fill(dimensions, T.Zero);

        public static DenseTensor<T> Fill<T>(ReadOnlySpan<int> dimensions, T value) where T : INumber<T>
        {
            var result = new DenseTensor<T>(dimensions);
            result.Fill(value);
            return result;
        }

        public static void Lerp(this DenseTensor<float> tensor1, DenseTensor<float> tensor2, float value)
        {
            TensorPrimitives.Lerp(tensor1.Buffer.Span, tensor2.Buffer.Span, value, tensor1.Buffer.Span);
        }


        public static void Lerp(this Span<float> span1, Span<float> span2, float value)
        {
            TensorPrimitives.Lerp(span1, span2, value, span1);
        }


        public static DenseTensor<T> CloneTensor<T>(this DenseTensor<T> source)
        {
            if (source is null)
                return null;

            return new DenseTensor<T>(new Memory<T>([.. source]), source.Dimensions, source.IsReversedStride);
        }


        public static DenseTensor<float> ResizeImage(this DenseTensor<float> sourceImage, int targetWidth, int targetHeight, ImageResizeMode resizeMode = ImageResizeMode.Stretch)
        {
            var cropX = 0;
            var cropY = 0;
            var croppedWidth = targetWidth;
            var croppedHeight = targetHeight;
            var channels = sourceImage.Dimensions[1];
            var sourceHeight = sourceImage.Dimensions[2];
            var sourceWidth = sourceImage.Dimensions[3];
            var destination = new DenseTensor<float>(new[] { 1, channels, targetHeight, targetWidth });
            if (resizeMode == ImageResizeMode.Crop)
            {
                var scaleX = (float)targetWidth / sourceWidth;
                var scaleY = (float)targetHeight / sourceHeight;
                var scaleFactor = Math.Max(scaleX, scaleY);
                croppedWidth = (int)(sourceWidth * scaleFactor);
                croppedHeight = (int)(sourceHeight * scaleFactor);
                cropX = Math.Abs(Math.Max((croppedWidth - targetWidth) / 2, 0));
                cropY = Math.Abs(Math.Max((croppedHeight - targetHeight) / 2, 0));
            }

            Parallel.For(0, channels, c =>
            {
                for (int h = 0; h < croppedHeight; h++)
                {
                    for (int w = 0; w < croppedWidth; w++)
                    {
                        // Map target pixel to input pixel
                        var y = h * (float)(sourceHeight - 1) / (croppedHeight - 1);
                        var x = w * (float)(sourceWidth - 1) / (croppedWidth - 1);

                        var y0 = (int)Math.Floor(y);
                        var x0 = (int)Math.Floor(x);
                        var y1 = Math.Min(y0 + 1, sourceHeight - 1);
                        var x1 = Math.Min(x0 + 1, sourceWidth - 1);

                        // Bilinear interpolation
                        var dy = y - y0;
                        var dx = x - x0;
                        var topLeft = sourceImage[0, c, y0, x0];
                        var topRight = sourceImage[0, c, y0, x1];
                        var bottomLeft = sourceImage[0, c, y1, x0];
                        var bottomRight = sourceImage[0, c, y1, x1];

                        var targetY = h - cropY;
                        var targetX = w - cropX;
                        if (targetX >= 0 && targetY >= 0 && targetY < targetHeight && targetX < targetWidth)
                        {
                            destination[0, c, targetY, targetX] =
                                    topLeft * (1 - dx) * (1 - dy) +
                                    topRight * dx * (1 - dy) +
                                    bottomLeft * (1 - dx) * dy +
                                    bottomRight * dx * dy;
                        }
                    }
                }
            });

            return destination;
        }


        public static DenseTensor<float> ResizeImageBicubic(this DenseTensor<float> sourceImage, int targetWidth, int targetHeight, ImageResizeMode resizeMode = ImageResizeMode.Stretch)
        {
            var cropX = 0;
            var cropY = 0;
            var croppedWidth = targetWidth;
            var croppedHeight = targetHeight;
            var channels = sourceImage.Dimensions[1];
            var sourceHeight = sourceImage.Dimensions[2];
            var sourceWidth = sourceImage.Dimensions[3];
            var destination = new DenseTensor<float>(new[] { 1, channels, targetHeight, targetWidth });
            if (resizeMode == ImageResizeMode.Crop)
            {
                var scaleX = (float)targetWidth / sourceWidth;
                var scaleY = (float)targetHeight / sourceHeight;
                var scaleFactor = Math.Max(scaleX, scaleY);
                croppedWidth = (int)(sourceWidth * scaleFactor);
                croppedHeight = (int)(sourceHeight * scaleFactor);
                cropX = Math.Abs(Math.Max((croppedWidth - targetWidth) / 2, 0));
                cropY = Math.Abs(Math.Max((croppedHeight - targetHeight) / 2, 0));
            }

            Parallel.For(0, channels, c =>
            {
                for (int h = 0; h < croppedHeight; h++)
                {
                    for (int w = 0; w < croppedWidth; w++)
                    {
                        float y = h * (float)(sourceHeight - 1) / (croppedHeight - 1);
                        float x = w * (float)(sourceWidth - 1) / (croppedWidth - 1);

                        int yInt = (int)Math.Floor(y);
                        int xInt = (int)Math.Floor(x);
                        float yFrac = y - yInt;
                        float xFrac = x - xInt;

                        float[] colVals = new float[4];

                        for (int i = -1; i <= 2; i++)
                        {
                            int yi = Math.Clamp(yInt + i, 0, sourceHeight - 1);
                            float[] rowVals = new float[4];

                            for (int j = -1; j <= 2; j++)
                            {
                                int xi = Math.Clamp(xInt + j, 0, sourceWidth - 1);
                                rowVals[j + 1] = sourceImage[0, c, yi, xi];
                            }

                            colVals[i + 1] = CubicInterpolate(rowVals[0], rowVals[1], rowVals[2], rowVals[3], xFrac);
                        }

                        var targetY = h - cropY;
                        var targetX = w - cropX;
                        if (targetX >= 0 && targetY >= 0 && targetY < targetHeight && targetX < targetWidth)
                        {
                            destination[0, c, h, w] = CubicInterpolate(colVals[0], colVals[1], colVals[2], colVals[3], yFrac);
                        }
                    }
                }
            });

            return destination;
        }


        private static float CubicInterpolate(float v0, float v1, float v2, float v3, float fraction)
        {
            float A = (-0.5f * v0) + (1.5f * v1) - (1.5f * v2) + (0.5f * v3);
            float B = (v0 * -1.0f) + (v1 * 2.5f) - (v2 * 2.0f) + (v3 * 0.5f);
            float C = (-0.5f * v0) + (0.5f * v2);
            float D = v1;
            return A * (fraction * fraction * fraction) + B * (fraction * fraction) + C * fraction + D;
        }
    }
}
