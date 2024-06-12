using OnnxStack.Common.Config;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace OnnxStack.Core.Config
{
    public class ConfigManager
    {
        private static string _configurationBaseDirectory = AppDomain.CurrentDomain.BaseDirectory;


        /// <summary>
        /// Sets the configuration location.
        /// </summary>
        /// <param name="baseDirectory">The base directory.</param>
        public static void SetConfiguration(string baseDirectory)
        {
            _configurationBaseDirectory = baseDirectory;
        }


        /// <summary>
        /// Loads the OnnxStackConfig configuration object from appsetting.json
        /// </summary>
        /// <returns>OnnxStackConfig object</returns>
        public static OnnxStackConfig LoadConfiguration()
        {
            return LoadConfiguration<OnnxStackConfig>();
        }


        /// <summary>
        /// Loads a custom IConfigSection object from appsetting.json
        /// </summary>
        /// <typeparam name="T">The custom IConfigSection class type, NOTE: json section name MUST match class name</typeparam>
        /// <returns>The deserialized custom configuration object</returns>
        public static T LoadConfiguration<T>(string sectionName = null, params JsonConverter[] converters) where T : class, IConfigSection
        {
            return LoadConfigurationSection<T>(sectionName, converters);
        }


        /// <summary>
        /// Loads a configuration section.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="converters">The converters.</param>
        /// <returns></returns>
        /// <exception cref="System.Exception">Failed to parse json element</exception>
        private static T LoadConfigurationSection<T>(string sectionName, params JsonConverter[] converters) where T : class, IConfigSection
        {
            var name = sectionName ?? typeof(T).Name;
            var serializerOptions = GetSerializerOptions(converters);
            var jsonDocument = GetJsonDocument(serializerOptions);
            var configElement = jsonDocument.RootElement.GetProperty(name);
            var configuration = configElement.Deserialize<T>(serializerOptions)
                ?? throw new Exception($"Failed to parse {name} json element");
            configuration.Initialize();
            return configuration;
        }


        /// <summary>
        /// Gets and loads the appsettings.json document and caches it
        /// </summary>
        /// <param name="serializerOptions">The serializer options.</param>
        /// <returns></returns>
        /// <exception cref="System.IO.FileNotFoundException"></exception>
        /// <exception cref="System.Exception">Failed to parse appsetting document</exception>
        private static JsonDocument GetJsonDocument(JsonSerializerOptions serializerOptions)
        {
            string appsettingStreamFile = GetAppSettingsFile();
            if (!File.Exists(appsettingStreamFile))
                throw new FileNotFoundException(appsettingStreamFile);

            using var appsettingStream = File.OpenRead(appsettingStreamFile);
            var appSettingsDocument = JsonSerializer.Deserialize<JsonDocument>(appsettingStream, serializerOptions)
                  ?? throw new Exception("Failed to parse appsetting document");

            return appSettingsDocument;
        }


        /// <summary>
        /// Gets the serializer options.
        /// </summary>
        /// <param name="jsonConverters">The json converters.</param>
        /// <returns>JsonSerializerOptions</returns>
        private static JsonSerializerOptions GetSerializerOptions(params JsonConverter[] jsonConverters)
        {
            var serializerOptions = new JsonSerializerOptions();
            serializerOptions.Converters.Add(new JsonStringEnumConverter());
            if (jsonConverters is not null)
            {
                foreach (var jsonConverter in jsonConverters)
                    serializerOptions.Converters.Add(jsonConverter);
            }
            return serializerOptions;
        }


        /// <summary>
        /// Saves the configuration.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public static void SaveConfiguration(OnnxStackConfig configuration)
        {
            SaveConfiguration<OnnxStackConfig>(configuration);
        }


        /// <summary>
        /// Saves the configuration.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="configuration">The configuration.</param>
        public static void SaveConfiguration<T>(T configuration) where T : class, IConfigSection
        {
            SaveConfiguration<T>(typeof(T).Name, configuration);
        }


        /// <summary>
        /// Saves the configuration.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="sectionName">Name of the section.</param>
        /// <param name="configuration">The configuration.</param>
        public static void SaveConfiguration<T>(string sectionName, T configuration) where T : class, IConfigSection
        {

            string appsettingStreamFile = GetAppSettingsFile();

            // Read In File
            Dictionary<string, object> appSettings;
            using (var appsettingReadStream = File.OpenRead(appsettingStreamFile))
                appSettings = JsonSerializer.Deserialize<Dictionary<string, object>>(appsettingReadStream, GetSerializerOptions());

            // Set Object
            appSettings[sectionName] = configuration;

            // Write out file
            var serializerOptions = GetSerializerOptions();
            serializerOptions.WriteIndented = true;
            using (var appsettingWriteStream = File.Open(appsettingStreamFile, FileMode.Create))
                JsonSerializer.Serialize(appsettingWriteStream, appSettings, serializerOptions);
        }


        /// <summary>
        /// Gets the application settings file.
        /// </summary>
        /// <returns></returns>
        private static string GetAppSettingsFile()
        {
            return Path.Combine(_configurationBaseDirectory, "appsettings.json");
        }
    }
}
