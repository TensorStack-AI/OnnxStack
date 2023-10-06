using Microsoft.AspNetCore.Mvc;
using Newtonsoft.Json;

namespace OnnxStack.Web.Models
{
    public class ModalResult : ActionResult
    {
        public ModalResult()
        {
        }

        public ModalResult(object data)
        {
            JsonData = JsonConvert.SerializeObject(data);
        }

        public string JsonData { get; set; }


        public override async Task ExecuteResultAsync(ActionContext context)
        {
            var response = context.HttpContext.Response;
            response.ContentType = "text/html";
            if (!string.IsNullOrEmpty(JsonData))
            {
                await response.WriteAsync(@"<script>(function () { $.modal.close(" + JsonData + "); })();</script>");
            }
            else
            {
                await response.WriteAsync(@"<script>(function () { $.modal.close(); })();</script>");
            }
        }

        public static ModalResult Close()
        {
            return new ModalResult();
        }

        public static ModalResult Close(object result)
        {
            return new ModalResult(result);
        }

        public static ModalResult Success(string message = "")
        {
            return new ModalResult(new { Success = true, Messsage = message });
        }

        public static ModalResult Error(string message = "")
        {
            return new ModalResult(new { Success = false, Messsage = message });
        }
    }
}
