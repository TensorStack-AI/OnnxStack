using Microsoft.ML.OnnxRuntime;

namespace OnnxStack.Core.Services
{
    public interface IOnnxModelAdaptaterService
    {
        void ApplyLowRankAdaptation(InferenceSession primarySession, InferenceSession loraSession);
    }
}