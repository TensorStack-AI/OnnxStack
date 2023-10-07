using OnnxStack.WebUI.Models;

namespace OnnxStack.Web.Hubs
{
    public interface IStableDiffusionClient
    {
        Task OnError(string error);
        Task OnMessage(string message);
        Task OnCanceled(string message);
        Task OnProgress(ProgressResult progress);
    }
    
}
