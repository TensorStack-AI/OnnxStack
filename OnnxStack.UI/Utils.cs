using OnnxStack.StableDiffusion.Config;
using OnnxStack.UI.Models;
using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

namespace OnnxStack.UI
{
    public static class Utils
    {
        public static string RandomString()
        {
            return Path.GetFileNameWithoutExtension(Path.GetRandomFileName());
        }

        public static BitmapImage CreateBitmap(byte[] imageBytes)
        {
            using (var memoryStream = new MemoryStream(imageBytes))
            {
                var image = new BitmapImage();
                image.BeginInit();
                image.CacheOption = BitmapCacheOption.OnLoad;
                image.StreamSource = memoryStream;
                image.EndInit();
                return image;
            }
        }

        public static async Task<bool> SaveImageFile(this ImageResult imageResult, string filename)
        {
            await Task.Run(() =>
            {
                var encoder = new PngBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(imageResult.Image));
                using (var fileStream = new FileStream(filename, FileMode.Create))
                {
                    encoder.Save(fileStream);
                }
            });
            return File.Exists(filename);
        }


        public static async Task<bool> SaveBlueprintFile(this ImageResult imageResult, string filename)
        {
            var serializerOptions = new JsonSerializerOptions
            {
                WriteIndented = true,
                Converters = { new JsonStringEnumConverter() }
            };
            using (var fileStream = new FileStream(filename, FileMode.Create))
            {
                await JsonSerializer.SerializeAsync(fileStream, imageResult, serializerOptions);
                return File.Exists(filename);
            }
        }

        public static SchedulerOptions ToSchedulerOptions(this SchedulerOptionsModel model)
        {
            return new SchedulerOptions
            {
                AlphaTransformType = model.AlphaTransformType,
                BetaEnd = model.BetaEnd,
                BetaStart = model.BetaStart,
                BetaSchedule = model.BetaSchedule,
                ClipSample = model.ClipSample,
                ClipSampleRange = model.ClipSampleRange,
                GuidanceScale = model.GuidanceScale,
                Height = model.Height,
                InferenceSteps = model.InferenceSteps,
                InitialNoiseLevel = model.InitialNoiseLevel,
                MaximumBeta = model.MaximumBeta,
                PredictionType = model.PredictionType,
                SampleMaxValue = model.SampleMaxValue,
                Seed = model.Seed,
                StepsOffset = model.StepsOffset,
                Width = model.Width,
                Strength = model.Strength,
                Thresholding = model.Thresholding,
                TimestepSpacing = model.TimestepSpacing,
                TrainedBetas = model.TrainedBetas,
                TrainTimesteps = model.TrainTimesteps,
                UseKarrasSigmas = model.UseKarrasSigmas,
                VarianceType = model.VarianceType
            };
        }

        public static SchedulerOptionsModel ToSchedulerOptionsModel(this SchedulerOptions model)
        {
            return new SchedulerOptionsModel
            {
                AlphaTransformType = model.AlphaTransformType,
                BetaEnd = model.BetaEnd,
                BetaStart = model.BetaStart,
                BetaSchedule = model.BetaSchedule,
                ClipSample = model.ClipSample,
                ClipSampleRange = model.ClipSampleRange,
                GuidanceScale = model.GuidanceScale,
                Height = model.Height,
                InferenceSteps = model.InferenceSteps,
                InitialNoiseLevel = model.InitialNoiseLevel,
                MaximumBeta = model.MaximumBeta,
                PredictionType = model.PredictionType,
                SampleMaxValue = model.SampleMaxValue,
                Seed = model.Seed,
                StepsOffset = model.StepsOffset,
                Width = model.Width,
                Strength = model.Strength,
                Thresholding = model.Thresholding,
                TimestepSpacing = model.TimestepSpacing,
                TrainedBetas = model.TrainedBetas,
                TrainTimesteps = model.TrainTimesteps,
                UseKarrasSigmas = model.UseKarrasSigmas,
                VarianceType = model.VarianceType
            };
        }

        public static PromptOptionsModel ToPromptOptionsModel(this PromptOptions promptOptions)
        {
            return new PromptOptionsModel
            {
                Prompt = promptOptions.Prompt,
                NegativePrompt = promptOptions.NegativePrompt,
                SchedulerType = promptOptions.SchedulerType
            };
        }

        public static void LogToWindow(string message)
        {
            System.Windows.Application.Current.Dispatcher.BeginInvoke(DispatcherPriority.Render, new Action(() =>
            {
                (System.Windows.Application.Current.MainWindow as MainWindow).UpdateOutputLog(message);
            }));
        }
    }
}
