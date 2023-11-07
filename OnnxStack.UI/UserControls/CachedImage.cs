using System;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

namespace OnnxStack.UI.UserControls
{
    public class CachedImage : Image
    {
        private static string _cachePath;

        private BitmapImage _bitmapImage;

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
                DependencyProperty.Register("ImageUrl", typeof(string), typeof(CachedImage), new PropertyMetadata((s,e) =>
                {
                    if (s is CachedImage cachedImage && e.NewValue is string imageUrl)
                        cachedImage.GetOrDownloadImage(imageUrl);
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
                DependencyProperty.Register("Placeholder", typeof(BitmapSource), typeof(CachedImage));

        /// <summary>
        /// Gets or downloads the image.
        /// </summary>
        /// <param name="imageUrl">The image URL.</param>
        public void GetOrDownloadImage(string imageUrl)
        {
            try
            {
                Source = Placeholder;
                if (string.IsNullOrEmpty(imageUrl))
                {
                    _bitmapImage = null;
                    return;
                }

                var filename = Path.GetFileName(imageUrl);
                var directory = Path.Combine(_cachePath, CacheName);
                var existingImage = Path.Combine(directory, filename);
                if (File.Exists(existingImage))
                {
                    LoadImage(existingImage);
                }
                else
                {
                    DownloadImage(imageUrl, existingImage);
                }
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
        public void LoadImage(string imageFile)
        {
            using (var fileStream = new FileStream(imageFile, FileMode.Open, FileAccess.Read))
            {
                _bitmapImage = new BitmapImage();
                _bitmapImage.BeginInit();
                _bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                _bitmapImage.StreamSource = fileStream;
                _bitmapImage.EndInit();
                Source = _bitmapImage;
            }
        }


        /// <summary>
        /// Downloads the image and saves it to the cache directory.
        /// </summary>
        /// <param name="source">The source.</param>
        /// <param name="destination">The destination.</param>
        public void DownloadImage(string source, string destination)
        {
            _bitmapImage = new BitmapImage();
            _bitmapImage.DownloadCompleted += (s, e) =>
            {
                if (_bitmapImage is null)
                    return;

                Source = _bitmapImage;
                Directory.CreateDirectory(Path.GetDirectoryName(destination));
                BitmapEncoder encoder = new PngBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(_bitmapImage));
                using (var fileStream = new FileStream(destination, FileMode.Create))
                {
                    encoder.Save(fileStream);
                }
            };
            _bitmapImage.BeginInit();
            _bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
            _bitmapImage.UriSource = new Uri(source);
            _bitmapImage.EndInit();
        }
    }
}
