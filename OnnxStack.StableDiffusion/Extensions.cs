using Microsoft.ML.OnnxRuntime;
using System.Linq;

namespace OnnxStack.StableDiffusion
{
    internal static class Extensions
    {
        public static T FirstElementAs<T>(this IDisposableReadOnlyCollection<DisposableNamedOnnxValue> collection)
        {
            if (collection is null || collection.Count == 0)
                return default;

            var element = collection.FirstOrDefault();
            if (element.Value is not T value)
                return default;

            return value;
        }

        public static T LastElementAs<T>(this IDisposableReadOnlyCollection<DisposableNamedOnnxValue> collection)
        {
            if (collection is null || collection.Count == 0)
                return default;

            var element = collection.LastOrDefault();
            if (element.Value is not T value)
                return default;

            return value;
        }
    }
}
