using OnnxStack.StableDiffusion.Enums;

namespace OnnxStack.Console
{
    internal static class Helpers
    {
        public static SchedulerType[] GetPipelineSchedulers(DiffuserPipelineType pipelineType)
        {
            return pipelineType switch
            {
                DiffuserPipelineType.StableDiffusion => new[]
                {
                    SchedulerType.LMS,
                    SchedulerType.Euler,
                    SchedulerType.EulerAncestral,
                    SchedulerType.DDPM, 
                    SchedulerType.DDIM,
                    SchedulerType.KDPM2 
                },
                DiffuserPipelineType.LatentConsistency => new[] 
                { 
                    SchedulerType.LCM 
                },
                _ => default
            };
        }
    }
}
