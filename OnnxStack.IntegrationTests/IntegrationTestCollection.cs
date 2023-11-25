namespace OnnxStack.IntegrationTests;

/// <summary>
/// All integration tests need to go in a single collection, so tests in different classes run sequentially and not in parallel.
/// </summary>
[CollectionDefinition("IntegrationTests")]
public class IntegrationTestCollection { }