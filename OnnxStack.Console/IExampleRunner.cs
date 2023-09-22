
namespace OnnxStack.Console
{
    /// <summary>
    /// Interface for Examples to inherit from, just add the interface to your class 
    /// and your Example will be included in the startup pipeline
    /// </summary>
    internal interface IExampleRunner
    {
        /// <summary>
        /// Gets the name of the Example.
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Gets the description of the Example.
        /// </summary>
        string Description { get; }


        /// <summary>
        /// The entry point for the Example, called when selected from the console
        /// </summary>
        /// <returns></returns>
        Task RunAsync();
    }
}
