using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Linq;

namespace OnnxStack.Core.Services
{
    public class OnnxModelAdaptaterService : IOnnxModelAdaptaterService
    {
        public void ApplyLowRankAdaptation(InferenceSession primarySession, InferenceSession loraSession)
        {
            // For simplicity, let's assume we will replace the weights of the first dense layer
            string layerName = "layer_name";

            // Get the current weights from the primary model
            var primaryInputName = primarySession.InputMetadata.Keys.First();
            var primaryInputTensor = primarySession.InputMetadata[primaryInputName];
            var primaryWeights = new float[primaryInputTensor.Dimensions.Product()];

            // Get the weights from the LoRA model
            var lraInputName = loraSession.InputMetadata.Keys.First();
            var lraInputTensor = loraSession.InputMetadata[lraInputName];
            var lraWeights = new float[lraInputTensor.Dimensions.Product()];

            // Apply LoRA (replace weights) this is where we will do the mutiplication of the weights
            // but for testing sake just brute for replacing
            Array.Copy(lraWeights, primaryWeights, Math.Min(primaryWeights.Length, lraWeights.Length));

            // Update the primary model tensor with the modified weights
            var tensor = new DenseTensor<float>(primaryWeights, primaryInputTensor.Dimensions.ToArray());
            var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor(primaryInputName, tensor) };

            // Will it run?
            primarySession.Run(inputs);
        }
    }

    public static class Ext
    {
        public static int Product(this int[] array)
        {
            int result = 1;
            foreach (int element in array)
            {
                result *= element;
            }
            return result;
        }
    }
}
