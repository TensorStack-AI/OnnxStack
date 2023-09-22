namespace OnnxStack.Common.Config
{
    /// <summary>
    /// Interface for implementing custom appsettings.json sections
    /// </summary>
    public interface IConfigSection
    {
        /// <summary>
        /// Perform any initialization, called directly after deserialization
        /// </summary>
        void Initialize();
    }
}
