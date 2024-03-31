using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Model;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Threading.Tasks;

namespace OnnxStack.Core.Image
{
    public static class Extensions
    {

        /// <summary>
        /// Converts to image mask.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns></returns>
        public static OnnxImage ToImageMask(this DenseTensor<float> imageTensor)
        {
            return new OnnxImage(imageTensor.FromMaskTensor());
        }


        /// <summary>
        /// Convert from single channle mask tensor to Rgba32 (Greyscale)
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns></returns>
        public static Image<Rgba32> FromMaskTensor(this DenseTensor<float> imageTensor)
        {
            var width = imageTensor.Dimensions[3];
            var height = imageTensor.Dimensions[2];
            using (var result = new Image<L8>(width, height))
            {
                for (var y = 0; y < height; y++)
                {
                    for (var x = 0; x < width; x++)
                    {
                        result[x, y] = new L8((byte)(imageTensor[0, 0, y, x] * 255.0f));
                    }
                }
                return result.CloneAs<Rgba32>();
            }
        }


        public static ResizeMode ToResizeMode(this ImageResizeMode resizeMode)
        {
            return resizeMode switch
            {
                ImageResizeMode.Stretch => ResizeMode.Stretch,
                _ => ResizeMode.Crop
            };
        }


        /// <summary>
        /// Splits the Tensor into 4 equal tiles.
        /// </summary>
        /// <param name="sourceTensor">The source tensor.</param>
        /// <returns>TODO: Optimize</returns>
        public static ImageTiles SplitImageTiles(this DenseTensor<float> sourceTensor, int overlap = 20)
        {
            var tileWidth = sourceTensor.Dimensions[3] / 2;
            var tileHeight = sourceTensor.Dimensions[2] / 2;
            return new ImageTiles(tileWidth, tileHeight, overlap,
                SplitImageTile(sourceTensor, 0, 0, tileHeight + overlap, tileWidth + overlap),
                SplitImageTile(sourceTensor, 0, tileWidth - overlap, tileHeight + overlap, tileWidth * 2),
                SplitImageTile(sourceTensor, tileHeight - overlap, 0, tileHeight * 2, tileWidth + overlap),
                SplitImageTile(sourceTensor, tileHeight - overlap, tileWidth - overlap, tileHeight * 2, tileWidth * 2));
        }


        /// <summary>
        /// Splits a tile from the source.
        /// </summary>
        /// <param name="source">The tensor.</param>
        /// <param name="startRow">The start row.</param>
        /// <param name="startCol">The start col.</param>
        /// <param name="endRow">The end row.</param>
        /// <param name="endCol">The end col.</param>
        /// <returns></returns>
        private static DenseTensor<float> SplitImageTile(DenseTensor<float> source, int startRow, int startCol, int endRow, int endCol)
        {
            int height = endRow - startRow;
            int width = endCol - startCol;
            int channels = source.Dimensions[1];
            var splitTensor = new DenseTensor<float>(new[] { 1, channels, height, width });
            Parallel.For(0, channels, (c) =>
            {
                Parallel.For(0, height, (i) =>
                {
                    Parallel.For(0, width, (j) =>
                    {
                        splitTensor[0, c, i, j] = source[0, c, startRow + i, startCol + j];
                    });
                });
            });
            return splitTensor;
        }


        /// <summary>
        /// Joins the tiles into a single Tensor.
        /// </summary>
        /// <param name="tiles">The tiles.</param>
        /// <returns>TODO: Optimize</returns>
        public static DenseTensor<float> JoinImageTiles(this ImageTiles tiles)
        {
            var totalWidth = tiles.Width * 2;
            var totalHeight = tiles.Height * 2;
            var channels = tiles.Tile1.Dimensions[1];
            var destination = new DenseTensor<float>(new[] { 1, channels, totalHeight, totalWidth });
            JoinImageTile(destination, tiles.Tile1, 0, 0, tiles.Height + tiles.Overlap, tiles.Width + tiles.Overlap);
            JoinImageTile(destination, tiles.Tile2, 0, tiles.Width - tiles.Overlap, tiles.Height + tiles.Overlap, totalWidth);
            JoinImageTile(destination, tiles.Tile3, tiles.Height - tiles.Overlap, 0, totalHeight, tiles.Width + tiles.Overlap);
            JoinImageTile(destination, tiles.Tile4, tiles.Height - tiles.Overlap, tiles.Width - tiles.Overlap, totalHeight, totalWidth);
            return destination;
        }


        /// <summary>
        /// Joins the tile to the destination tensor.
        /// </summary>
        /// <param name="destination">The destination.</param>
        /// <param name="tile">The tile.</param>
        /// <param name="startRow">The start row.</param>
        /// <param name="startCol">The start col.</param>
        /// <param name="endRow">The end row.</param>
        /// <param name="endCol">The end col.</param>
        private static void JoinImageTile(DenseTensor<float> destination, DenseTensor<float> tile, int startRow, int startCol, int endRow, int endCol)
        {
            int height = endRow - startRow;
            int width = endCol - startCol;
            int channels = tile.Dimensions[1];
            Parallel.For(0, channels, (c) =>
            {
                Parallel.For(0, height, (i) =>
                {
                    Parallel.For(0, width, (j) =>
                    {
                        var value = tile[0, c, i, j];
                        var existing = destination[0, c, startRow + i, startCol + j];
                        if (existing > 0)
                        {
                            // Blend ovelap
                            value = (existing + value) / 2f;
                        }
                        destination[0, c, startRow + i, startCol + j] = value;
                    });
                });
            });
        }
    }

    public enum ImageNormalizeType
    {
        ZeroToOne = 0,
        OneToOne = 1
    }

    public enum ImageResizeMode
    {
        Crop = 0,
        Stretch = 1
    }
}
