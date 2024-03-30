using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxStack.Core.Model
{
    public record ImageTiles(int Width, int Height, int Overlap, DenseTensor<float> Tile1, DenseTensor<float> Tile2, DenseTensor<float> Tile3, DenseTensor<float> Tile4);
}
