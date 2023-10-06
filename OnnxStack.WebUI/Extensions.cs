using Microsoft.AspNetCore.Mvc.Rendering;

namespace OnnxStack.WebUI
{
	public static class Extensions
	{
        public static MvcForm AjaxBeginFormModal(this IHtmlHelper htmlHelper, string handler)
        {
            return htmlHelper.AjaxBeginForm(new AjaxOptions
            {
                Url = $"?handler={handler}",
                HttpMethod = "POST",
                UpdateTargetId = "simplemodal-data",
                InsertionMode = InsertionMode.Replace
            });
        }

        public static MvcForm AjaxBeginFormModal(this IHtmlHelper htmlHelper, string page, string handler)
		{
			return htmlHelper.AjaxBeginForm(new AjaxOptions
			{
				Url = $"{page}?handler={handler}",
				HttpMethod = "POST",
				UpdateTargetId = "simplemodal-data",
				InsertionMode = InsertionMode.Replace
			});
		}
	}
}
