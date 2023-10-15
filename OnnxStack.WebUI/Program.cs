using AspNetCore.Unobtrusive.Ajax;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using OnnxStack.Core;
using OnnxStack.Web.Hubs;
using Services;
using System.Text.Json.Serialization;

namespace OnnxStack.WebUI
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Add services to the container.
            builder.Services.AddRazorPages();
            builder.Services.AddUnobtrusiveAjax();
            builder.Services.AddSignalR(options =>
            {
                options.EnableDetailedErrors = true;
                options.MaximumReceiveMessageSize = 5242880;
            }).AddJsonProtocol(options =>
            {
                options.PayloadSerializerOptions.Converters.Add(new JsonStringEnumConverter());
            });

            builder.Services.AddOnnxStackStableDiffusion();
            builder.Services.AddSingleton<IFileService, FileService>();
            builder.Services.AddSingleton((s) => s.GetService<IServer>().Features.Get<IServerAddressesFeature>());

            var app = builder.Build();

            // Configure the HTTP request pipeline.
            if (!app.Environment.IsDevelopment())
            {
                app.UseExceptionHandler("/Error");
            }
            app.UseStaticFiles();

            app.UseUnobtrusiveAjax();

            app.UseRouting();

            app.UseAuthorization();

            app.MapRazorPages();

            app.MapHub<StableDiffusionHub>(nameof(StableDiffusionHub));

            app.Run();
        }
    }
}