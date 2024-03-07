using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxStack.Core.Model
{
    public record ImageTiles(DenseTensor<float> Tile1, DenseTensor<float> Tile2, DenseTensor<float> Tile3, DenseTensor<float> Tile4);
}
