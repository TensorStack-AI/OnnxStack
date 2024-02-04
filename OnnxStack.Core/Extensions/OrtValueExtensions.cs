using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Model;
using System;

namespace OnnxStack.Core
{
    public static class OrtValueExtensions
    {
        /// <summary>
        /// Creates and OrtValue form the DenseTensor and NodeMetaData provided
        /// TODO: Optimization
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="metadata">The metadata.</param>
        /// <returns></returns>
        public static OrtValue ToOrtValue(this DenseTensor<float> tensor, OnnxNamedMetadata metadata)
        {
            var dimensions = tensor.Dimensions.ToLong();
            return metadata.Value.ElementDataType switch
            {
                TensorElementType.Int64 => OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Buffer.ToLong(), dimensions),
                TensorElementType.Float16 => OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Buffer.ToFloat16(), dimensions),
                TensorElementType.BFloat16 => OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Buffer.ToBFloat16(), dimensions),
                _ => OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Buffer, dimensions)
            };
        }


        /// <summary>
        /// Converts DenseTensor<string> to OrtValue.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns></returns>
        public static OrtValue ToOrtValue(this DenseTensor<string> tensor, OnnxNamedMetadata metadata)
        {
            return OrtValue.CreateFromStringTensor(tensor);
        }


        /// <summary>
        /// Converts DenseTensor<int> to OrtValue.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns></returns>
        public static OrtValue ToOrtValue(this DenseTensor<int> tensor, OnnxNamedMetadata metadata)
        {
            return OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Buffer, tensor.Dimensions.ToLong());
        }


        /// <summary>
        /// Converts DenseTensor<int> to OrtValue.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns></returns>
        public static OrtValue ToOrtValue(this DenseTensor<long> tensor, OnnxNamedMetadata metadata)
        {
            return OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Buffer, tensor.Dimensions.ToLong());
        }


        /// <summary>
        /// Converts DenseTensor<double> to OrtValue.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="metadata">The metadata.</param>
        /// <returns></returns>
        public static OrtValue ToOrtValue(this DenseTensor<double> tensor, OnnxNamedMetadata metadata)
        {
            return OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Buffer, tensor.Dimensions.ToLong());
        }


        /// <summary>
        /// Converts DenseTensor<bool> to OrtValue.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="metadata">The metadata.</param>
        /// <returns></returns>
        public static OrtValue ToOrtValue(this DenseTensor<bool> tensor, OnnxNamedMetadata metadata)
        {
            return OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Buffer, tensor.Dimensions.ToLong());
        }


        /// <summary>
        /// Creates and allocates the output tensors buffer.
        /// </summary>
        /// <param name="metadata">The metadata.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        public static OrtValue CreateOutputBuffer(this OnnxNamedMetadata metadata, ReadOnlySpan<int> dimensions)
        {
            return OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, metadata.Value.ElementDataType, dimensions.ToLong());
        }


        /// <summary>
        /// Converts to DenseTensor<float>.
        /// TODO: Optimization
        /// </summary>
        /// <param name="ortValue">The ort value.</param>
        /// <returns></returns>
        public static DenseTensor<float> ToDenseTensor(this OrtValue ortValue)
        {
            var typeInfo = ortValue.GetTensorTypeAndShape();
            var dimensions = typeInfo.Shape.ToInt();
            return typeInfo.ElementDataType switch
            {
                TensorElementType.Float16 => new DenseTensor<float>(ortValue.GetTensorDataAsSpan<Float16>().ToFloat(), dimensions),
                TensorElementType.BFloat16 => new DenseTensor<float>(ortValue.GetTensorDataAsSpan<BFloat16>().ToFloat(), dimensions),
                _ => new DenseTensor<float>(ortValue.GetTensorDataAsSpan<float>().ToArray(), dimensions)
            };
        }


        /// <summary>
        /// Converts to array.
        /// TODO: Optimization
        /// </summary>
        /// <param name="ortValue">The ort value.</param>
        /// <returns></returns>
        public static float[] ToArray(this OrtValue ortValue)
        {
            var typeInfo = ortValue.GetTensorTypeAndShape();
            return typeInfo.ElementDataType switch
            {
                TensorElementType.Float16 => ortValue.GetTensorDataAsSpan<Float16>().ToFloat().ToArray(),
                TensorElementType.BFloat16 => ortValue.GetTensorDataAsSpan<BFloat16>().ToFloat().ToArray(),
                _ => ortValue.GetTensorDataAsSpan<float>().ToArray()
            };
        }


        public static T[] ToArray<T>(this OrtValue ortValue) where T : unmanaged
        {
            return ortValue.GetTensorDataAsSpan<T>().ToArray();
        }



        /// <summary>
        /// Converts to float16.
        /// TODO: Optimization
        /// </summary>
        /// <param name="inputMemory">The input memory.</param>
        /// <returns></returns>
        private static Memory<Float16> ToFloat16(this Memory<float> inputMemory)
        {
            var elementCount = inputMemory.Length;
            var floatArray = new Float16[inputMemory.Length];
            for (int i = 0; i < elementCount; i++)
                floatArray[i] = (Float16)inputMemory.Span[i];

            return floatArray.AsMemory();
        }


        /// <summary>
        /// Converts to BFloat16.
        /// TODO: Optimization
        /// </summary>
        /// <param name="inputMemory">The input memory.</param>
        /// <returns></returns>
        private static Memory<BFloat16> ToBFloat16(this Memory<float> inputMemory)
        {
            var elementCount = inputMemory.Length;
            var floatArray = new BFloat16[inputMemory.Length];
            for (int i = 0; i < elementCount; i++)
                floatArray[i] = (BFloat16)inputMemory.Span[i];

            return floatArray.AsMemory();
        }


        /// <summary>
        /// Converts to float.
        /// TODO: Optimization
        /// </summary>
        /// <param name="inputMemory">The input memory.</param>
        /// <returns></returns>
        private static Memory<float> ToFloat(this ReadOnlySpan<Float16> inputMemory)
        {
            var elementCount = inputMemory.Length;
            var floatArray = new float[elementCount];
            for (int i = 0; i < elementCount; i++)
                floatArray[i] = (float)inputMemory[i];

            return floatArray.AsMemory();
        }


        /// <summary>
        /// Converts to float.
        /// TODO: Optimization
        /// </summary>
        /// <param name="inputMemory">The input memory.</param>
        /// <returns></returns>
        private static Memory<float> ToFloat(this ReadOnlySpan<BFloat16> inputMemory)
        {
            var elementCount = inputMemory.Length;
            var floatArray = new float[elementCount];
            for (int i = 0; i < elementCount; i++)
                floatArray[i] = (float)inputMemory[i];

            return floatArray.AsMemory();
        }


        /// <summary>
        /// Converts to long.
        /// </summary>
        /// <param name="inputMemory">The input memory.</param>
        /// <returns></returns>
        private static Memory<long> ToLong(this Memory<float> inputMemory)
        {
            return Array.ConvertAll(inputMemory.ToArray(), Convert.ToInt64).AsMemory();
        }
    }
}
