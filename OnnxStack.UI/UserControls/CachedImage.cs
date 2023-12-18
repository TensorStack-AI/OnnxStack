using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

namespace OnnxStack.UI.UserControls
{
    public class CachedImage : Image
    {
        private static string _cachePath;

        static CachedImage()
        {
            DefaultStyleKeyProperty.OverrideMetadata(typeof(CachedImage), new FrameworkPropertyMetadata(typeof(CachedImage)));
            _cachePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, ".cache");
            if (Directory.Exists(_cachePath))
                Directory.CreateDirectory(_cachePath);
        }

        public string ImageUrl
        {
            get { return (string)GetValue(ImageUrlProperty); }
            set { SetValue(ImageUrlProperty, value); }
        }
        public static readonly DependencyProperty ImageUrlProperty =
                DependencyProperty.Register("ImageUrl", typeof(string), typeof(CachedImage), new PropertyMetadata(async (s, e) =>
                {
                    if (s is CachedImage cachedImage)
                        await cachedImage.GetOrDownloadImage(e.NewValue as string);
                }));

        public string CacheName
        {
            get { return (string)GetValue(CacheNameProperty); }
            set { SetValue(CacheNameProperty, value); }
        }
        public static readonly DependencyProperty CacheNameProperty =
                 DependencyProperty.Register("CacheName", typeof(string), typeof(CachedImage));

        public BitmapSource Placeholder
        {
            get { return (BitmapSource)GetValue(PlaceholderProperty); }
            set { SetValue(PlaceholderProperty, value); }
        }
        public static readonly DependencyProperty PlaceholderProperty =
                DependencyProperty.Register("Placeholder", typeof(BitmapSource), typeof(CachedImage), new PropertyMetadata(async (s, e) =>
                {
                    if (s is CachedImage cachedImage && cachedImage.Source is null)
                        await cachedImage.GetOrDownloadImage(cachedImage.ImageUrl);
                }));


        /// <summary>
        /// Gets or downloads the image.
        /// </summary>
        /// <param name="imageUrl">The image URL.</param>
        public async Task GetOrDownloadImage(string imageUrl)
        {
            try
            {
                if (string.IsNullOrEmpty(imageUrl))
                {
                    Source = Placeholder;
                    return;
                }

                var filename = Path.GetFileName(imageUrl);
                var directory = Path.Combine(_cachePath, CacheName ?? ".default");
                var existingImage = Path.Combine(directory, filename);

                Source = File.Exists(existingImage)
                    ? await LoadImage(existingImage)
                    : await DownloadImage(imageUrl, existingImage);
            }
            catch (Exception)
            {
                Source = Placeholder;
            }
        }


        /// <summary>
        /// Loads the image from the cahe directory.
        /// </summary>
        /// <param name="imageFile">The image file.</param>
        private static async Task<BitmapImage> LoadImage(string imageFile)
        {
            var tcs = new TaskCompletionSource<BitmapImage>();
            try
            {
                using (var fileStream = new FileStream(imageFile, FileMode.Open, FileAccess.Read))
                {
                    var bitmapImage = new BitmapImage();
                    bitmapImage.BeginInit();
                    bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                    bitmapImage.StreamSource = fileStream;
                    bitmapImage.EndInit();
                    bitmapImage.Freeze();
                    tcs.SetResult(bitmapImage);
                }
            }
            catch (Exception ex)
            {
                tcs.SetException(ex);
            }
            return await tcs.Task;
        }


        /// <summary>
        /// Downloads the image and saves it to the cache directory.
        /// </summary>
        /// <param name="source">The source.</param>
        /// <param name="destination">The destination.</param>
        private static async Task<BitmapImage> DownloadImage(string source, string destination)
        {
            var bitmapImage = new BitmapImage();
            using (var httpClient = new HttpClient())
            using (var imageStream = new MemoryStream(await httpClient.GetByteArrayAsync(source)))
            {
                var imagebytes = await httpClient.GetByteArrayAsync(source);
                bitmapImage.BeginInit();
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapImage.StreamSource = imageStream;
                bitmapImage.EndInit();
                bitmapImage.Freeze();

                var encoder = new PngBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(bitmapImage));
                Directory.CreateDirectory(Path.GetDirectoryName(destination));
                using (var fileStream = new FileStream(destination, FileMode.CreateNew))
                {
                    encoder.Save(fileStream);
                }
                return bitmapImage;
            }
        }
    }

}
