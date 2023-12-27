using System.Collections.Generic;

namespace OnnxStack.Core.Config
{
    public class OnnxModelEqualityComparer : IEqualityComparer<IOnnxModel>
    {
        public bool Equals(IOnnxModel x, IOnnxModel y)
        {
            return x != null && y != null && x.Name == y.Name;
        }

        public int GetHashCode(IOnnxModel obj)
        {
            return obj?.Name?.GetHashCode() ?? 0;
        }
    }
}
