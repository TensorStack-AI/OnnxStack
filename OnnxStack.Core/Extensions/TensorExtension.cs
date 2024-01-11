using Microsoft.ML.OnnxRuntime.Tensors;
using System;

namespace OnnxStack.Core
{
    public static class TensorExtension
    {
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
                for (int j = 0; j < tensor1.Dimensions[1]; j++)
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
    }
}
