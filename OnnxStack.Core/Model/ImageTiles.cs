using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Image;
using System.Threading.Tasks;

namespace OnnxStack.Core.Model
{
    public record ImageTiles
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ImageTiles"/> class.
        /// </summary>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="overlap">The overlap.</param>
        /// <param name="tile1">The tile1.</param>
        /// <param name="tile2">The tile2.</param>
        /// <param name="tile3">The tile3.</param>
        /// <param name="tile4">The tile4.</param>
        public ImageTiles(int width, int height, TileMode tileMode, int overlap, DenseTensor<float> tile1, DenseTensor<float> tile2, DenseTensor<float> tile3, DenseTensor<float> tile4)
        {
            Width = width;
            Height = height;
            Overlap = overlap;
            TileMode = tileMode;
            Tile1 = tile1;
            Tile2 = tile2;
            Tile3 = tile3;
            Tile4 = tile4;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ImageTiles"/> class.
        /// </summary>
        /// <param name="sourceTensor">The source tensor.</param>
        /// <param name="overlap">The overlap.</param>
        public ImageTiles(DenseTensor<float> sourceTensor, TileMode tileMode, int overlap = 16)
        {
            Overlap = overlap;
            TileMode = tileMode;
            Width = sourceTensor.Dimensions[3] / 2;
            Height = sourceTensor.Dimensions[2] / 2;
            Tile1 = SplitImageTile(sourceTensor, 0, 0, Height + overlap, Width + overlap);
            Tile2 = SplitImageTile(sourceTensor, 0, Width - overlap, Height + overlap, Width * 2);
            Tile3 = SplitImageTile(sourceTensor, Height - overlap, 0, Height * 2, Width + overlap);
            Tile4 = SplitImageTile(sourceTensor, Height - overlap, Width - overlap, Height * 2, Width * 2);
        }

        public int Width { get; init; }
        public int Height { get; init; }
        public int Overlap { get; init; }
        public TileMode TileMode { get; set; }
        public DenseTensor<float> Tile1 { get; init; }
        public DenseTensor<float> Tile2 { get; init; }
        public DenseTensor<float> Tile3 { get; init; }
        public DenseTensor<float> Tile4 { get; init; }


        /// <summary>
        /// Joins the tiles into a single ImageTensor.
        /// </summary>
        /// <returns>ImageTensor.</returns>
        public DenseTensor<float> JoinTiles()
        {
            var totalWidth = Width * 2;
            var totalHeight = Height * 2;
            var channels = Tile1.Dimensions[1];
            var destination = new DenseTensor<float>(new[] { 1, channels, totalHeight, totalWidth });
            JoinTileClip(destination, Tile1, 0, 0, Height + Overlap, Width + Overlap);
            JoinTileClip(destination, Tile2, 0, Width - Overlap, Height + Overlap, totalWidth);
            JoinTileClip(destination, Tile3, Height - Overlap, 0, totalHeight, Width + Overlap);
            JoinTileClip(destination, Tile4, Height - Overlap, Width - Overlap, totalHeight, totalWidth);
            return destination;
        }


        /// <summary>
        /// Joins the tile to a destination.
        /// </summary>
        /// <param name="destination">The destination.</param>
        /// <param name="tile">The tile.</param>
        /// <param name="startRow">The start row.</param>
        /// <param name="startCol">The start col.</param>
        /// <param name="endRow">The end row.</param>
        /// <param name="endCol">The end col.</param>
        public void JoinTile(DenseTensor<float> destination, DenseTensor<float> tile, int startRow, int startCol, int endRow, int endCol)
        {
            switch (TileMode)
            {
                case TileMode.Blend:
                    JoinTileBlend(destination, tile, startRow, startCol, endRow, endCol);
                    break;
                case TileMode.Clip:
                    JoinTileClip(destination, tile, startRow, startCol, endRow, endCol);
                    break;
                case TileMode.ClipBlend:
                    JoinTileClipBlend(destination, tile, startRow, startCol, endRow, endCol);
                    break;
                case TileMode.Overlap:
                case TileMode.None:
                default:
                    JoinTileOverlap(destination, tile, startRow, startCol, endRow, endCol);
                    break;
            }
        }


        /// <summary>
        /// Joins the tiles overlapping edges.
        /// </summary>
        /// <param name="destination">The destination.</param>
        /// <param name="tile">The tile.</param>
        /// <param name="startRow">The start row.</param>
        /// <param name="startCol">The start col.</param>
        /// <param name="endRow">The end row.</param>
        /// <param name="endCol">The end col.</param>
        private static void JoinTileOverlap(DenseTensor<float> destination, DenseTensor<float> tile, int startRow, int startCol, int endRow, int endCol)
        {
            int height = endRow - startRow;
            int width = endCol - startCol;
            int channels = tile.Dimensions[1];
            Parallel.For(0, channels, (c) =>
            {
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        destination[0, c, startRow + i, startCol + j] = tile[0, c, i, j];
                    }
                }
            });
        }


        /// <summary>
        /// Joins the tiles blending the overlapped edges.
        /// </summary>
        /// <param name="destination">The destination.</param>
        /// <param name="tile">The tile.</param>
        /// <param name="startRow">The start row.</param>
        /// <param name="startCol">The start col.</param>
        /// <param name="endRow">The end row.</param>
        /// <param name="endCol">The end col.</param>
        private void JoinTileBlend(DenseTensor<float> destination, DenseTensor<float> tile, int startRow, int startCol, int endRow, int endCol)
        {
            int height = endRow - startRow;
            int width = endCol - startCol;
            int channels = tile.Dimensions[1];
            Parallel.For(0, channels, (c) =>
            {
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        var value = tile[0, c, i, j];
                        var existing = destination[0, c, startRow + i, startCol + j];
                        if (existing > 0)
                        {
                            value = (existing + value) / 2f;
                        }
                        destination[0, c, startRow + i, startCol + j] = value;
                    }
                }
            });
        }


        /// <summary>
        /// Joins the tiles clipping the overlapped edges.
        /// </summary>
        /// <param name="destination">The destination.</param>
        /// <param name="tile">The tile.</param>
        /// <param name="startRow">The start row.</param>
        /// <param name="startCol">The start col.</param>
        /// <param name="endRow">The end row.</param>
        /// <param name="endCol">The end col.</param>
        private void JoinTileClip(DenseTensor<float> destination, DenseTensor<float> tile, int startRow, int startCol, int endRow, int endCol)
        {
            int height = endRow - startRow;
            int width = endCol - startCol;
            int channels = tile.Dimensions[1];
            Parallel.For(0, channels, (c) =>
            {
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        if (startRow > 0 && i < Overlap)
                            continue;
                        if (startCol > 0 && j < Overlap)
                            continue;
                        if (startRow == 0 && i > (endRow - Overlap))
                            continue;
                        if (startCol == 0 && j > (endCol - Overlap))
                            continue;

                        var existing = destination[0, c, startRow + i, startCol + j];
                        if (existing > 0)
                            continue;

                        destination[0, c, startRow + i, startCol + j] = tile[0, c, i, j];
                    }
                }
            });
        }


        /// <summary>
        /// Joins the tiles clipping  and blending the overlapped edges.
        /// </summary>
        /// <param name="destination">The destination.</param>
        /// <param name="tile">The tile.</param>
        /// <param name="startRow">The start row.</param>
        /// <param name="startCol">The start col.</param>
        /// <param name="endRow">The end row.</param>
        /// <param name="endCol">The end col.</param>
        private void JoinTileClipBlend(DenseTensor<float> destination, DenseTensor<float> tile, int startRow, int startCol, int endRow, int endCol)
        {
            int clip = Overlap / 2;
            int height = endRow - startRow;
            int width = endCol - startCol;
            int channels = tile.Dimensions[1];
            Parallel.For(0, channels, (c) =>
            {
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        if (startRow > 0 && i < clip)
                            continue;
                        if (startCol > 0 && j < clip)
                            continue;
                        if (startRow == 0 && i > (endRow - clip))
                            continue;
                        if (startCol == 0 && j > (endCol - clip))
                            continue;

                        var value = tile[0, c, i, j];
                        var existing = destination[0, c, startRow + i, startCol + j];
                        if (existing > 0)
                            value = (existing + value) / 2f;

                        destination[0, c, startRow + i, startCol + j] = value;
                    }
                }
            });
        }


        /// <summary>
        /// Splits the image tile.
        /// </summary>
        /// <param name="source">The source.</param>
        /// <param name="startRow">The start row.</param>
        /// <param name="startCol">The start col.</param>
        /// <param name="endRow">The end row.</param>
        /// <param name="endCol">The end col.</param>
        /// <returns>ImageTensor.</returns>
        private static DenseTensor<float> SplitImageTile(DenseTensor<float> source, int startRow, int startCol, int endRow, int endCol)
        {
            int height = endRow - startRow;
            int width = endCol - startCol;
            int channels = source.Dimensions[1];
            var splitTensor = new DenseTensor<float>(new[] { 1, channels, height, width });
            Parallel.For(0, channels, (c) =>
            {
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        splitTensor[0, c, i, j] = source[0, c, startRow + i, startCol + j];
                    }
                }
            });
            return splitTensor;
        }

    }

}
