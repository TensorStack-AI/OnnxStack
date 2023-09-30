using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.StableDiffusion.Config;
using System;
using System.Linq;

namespace OnnxStack.StableDiffusion.Helpers
{
    public static class TensorHelper
    {
        public static DenseTensor<T> CreateTensor<T>(T[] data, ReadOnlySpan<int> dimensions)
        {
            return new DenseTensor<T>(data, dimensions);
        }

        public static DenseTensor<float> DivideTensorByFloat(this DenseTensor<float> data, float value, ReadOnlySpan<int> dimensions)
        {
            var divTensor = new DenseTensor<float>(dimensions);
            for (int i = 0; i < data.Length; i++)
            {
                divTensor.SetValue(i, data.GetValue(i) / value);
            }
            return divTensor;
        }

        public static DenseTensor<float> DivideTensorByFloat(this DenseTensor<float> data, float value)
        {
            var divTensor = new DenseTensor<float>(data.Dimensions);
            for (int i = 0; i < data.Length; i++)
            {
                divTensor.SetValue(i, data.GetValue(i) / value);
            }
            return divTensor;
        }

        public static DenseTensor<float> MultipleTensorByFloat(this DenseTensor<float> data, float value)
        {
            var mullTensor = new DenseTensor<float>(data.Dimensions);
            for (int i = 0; i < data.Length; i++)
            {
                mullTensor.SetValue(i, data.GetValue(i) * value);
            }
            return mullTensor;
        }

        public static DenseTensor<float> AddTensors(this DenseTensor<float> sample, DenseTensor<float> sumTensor)
        {
            var addTensor = new DenseTensor<float>(sample.Dimensions);
            for (var i = 0; i < sample.Length; i++)
            {
                addTensor.SetValue(i, sample.GetValue(i) + sumTensor.GetValue(i));
            }
            return addTensor;
        }

        public static Tuple<DenseTensor<float>, DenseTensor<float>> SplitTensor(this DenseTensor<float> tensorToSplit, ReadOnlySpan<int> dimensions, int scaledHeight, int scaledWidth)
        {
            var tensor1 = new DenseTensor<float>(dimensions);
            var tensor2 = new DenseTensor<float>(dimensions);
            for (int i = 0; i < 1; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    for (int k = 0; k < scaledHeight; k++)
                    {
                        for (int l = 0; l < scaledWidth; l++)
                        {
                            tensor1[i, j, k, l] = tensorToSplit[i, j, k, l];
                            tensor2[i, j, k, l] = tensorToSplit[i, j + 4, k, l];
                        }
                    }
                }
            }
            return new Tuple<DenseTensor<float>, DenseTensor<float>>(tensor1, tensor2);
        }

        public static DenseTensor<float> SumTensors(this DenseTensor<float>[] tensorArray, ReadOnlySpan<int> dimensions)
        {
            var sumTensor = new DenseTensor<float>(dimensions);
            for (int m = 0; m < tensorArray.Length; m++)
            {
                var tensorToSum = tensorArray[m];
                for (var i = 0; i < tensorToSum.Length; i++)
                {
                    sumTensor.SetValue(i, sumTensor.GetValue(i) + tensorToSum.GetValue(i));
                }
            }
            return sumTensor;
        }

        public static DenseTensor<float> Duplicate(this DenseTensor<float> data, ReadOnlySpan<int> dimensions)
        {
            var dupTensor = data.Concat(data).ToArray();
            return CreateTensor(dupTensor, dimensions);
        }

        public static DenseTensor<float> SubtractTensors(this DenseTensor<float> sample, DenseTensor<float> subTensor, ReadOnlySpan<int> dimensions)
        {
            var result = new DenseTensor<float>(dimensions);
            for (var i = 0; i < sample.Length; i++)
            {
                result.SetValue(i, sample.GetValue(i) - subTensor.GetValue(i));
            }
            return result;
        }

        public static DenseTensor<float> SubtractTensors(this DenseTensor<float> sample, DenseTensor<float> subTensor)
        {
            return sample.SubtractTensors(subTensor, sample.Dimensions);
        }



        /// <summary>
        /// Reorders the tensor.
        /// </summary>
        /// <param name="inputTensor">The input tensor.</param>
        /// <returns></returns>
        public static DenseTensor<float> ReorderTensor(this DenseTensor<float> inputTensor, ReadOnlySpan<int> dimensions)
        {
            //reorder from batch channel height width to batch height width channel
            var inputImagesTensor = new DenseTensor<float>(dimensions);
            for (int y = 0; y < inputTensor.Dimensions[2]; y++)
            {
                for (int x = 0; x < inputTensor.Dimensions[3]; x++)
                {
                    inputImagesTensor[0, y, x, 0] = inputTensor[0, 0, y, x];
                    inputImagesTensor[0, y, x, 1] = inputTensor[0, 1, y, x];
                    inputImagesTensor[0, y, x, 2] = inputTensor[0, 2, y, x];
                }
            }
            return inputImagesTensor;
        }


        public static DenseTensor<float> PerformGuidance(this DenseTensor<float> noisePred, DenseTensor<float> noisePredText, double guidanceScale)
        {
            for (int i = 0; i < noisePred.Dimensions[0]; i++)
            {
                for (int j = 0; j < noisePred.Dimensions[1]; j++)
                {
                    for (int k = 0; k < noisePred.Dimensions[2]; k++)
                    {
                        for (int l = 0; l < noisePred.Dimensions[3]; l++)
                        {
                            noisePred[i, j, k, l] = noisePred[i, j, k, l] + (float)guidanceScale * (noisePredText[i, j, k, l] - noisePred[i, j, k, l]);
                        }
                    }
                }
            }
            return noisePred;
        }


        public static DenseTensor<float> Clip(this DenseTensor<float> tensor, float minValue, float maxValue)
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                tensor.SetValue(i, Math.Clamp(tensor.GetValue(i), minValue, maxValue));
            }
            return tensor;
        }


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
