global using Xunit;

// need all tests to run one at a time sequentially to not overwhelm the GPU
[assembly: CollectionBehavior(DisableTestParallelization = true)]