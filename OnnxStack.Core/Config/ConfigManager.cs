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
        public static T LoadConfiguration<T>(params JsonConverter[] converters) where T : class, IConfigSection
        {
            return LoadConfigurationSection<T>(converters);
        }


        /// <summary>
        /// Loads a configuration section.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="converters">The converters.</param>
        /// <returns></returns>
        /// <exception cref="System.Exception">Failed to parse json element</exception>
        private static T LoadConfigurationSection<T>(params JsonConverter[] converters) where T : class, IConfigSection
        {
            var serializerOptions = GetSerializerOptions(converters);
            var jsonDocument = GetJsonDocument(serializerOptions);
            var configElement = jsonDocument.RootElement.GetProperty(typeof(T).Name);
            var configuration = configElement.Deserialize<T>(serializerOptions)
                ?? throw new Exception($"Failed to parse {typeof(T).Name} json element");
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

        public static void SaveConfiguration(OnnxStackConfig configuration)
        {
            SaveConfiguration<OnnxStackConfig>(configuration);
        }

        public static void SaveConfiguration<T>(T configuration)
        {
          
            string appsettingStreamFile = GetAppSettingsFile();

            // Read In File
            Dictionary<string, object> appSettings;
            using (var appsettingReadStream = File.OpenRead(appsettingStreamFile))
                appSettings = JsonSerializer.Deserialize<Dictionary<string, object>>(appsettingReadStream, GetSerializerOptions());

            // Set Object
            appSettings[typeof(T).Name] = configuration;

            // Write out file
            var serializerOptions = GetSerializerOptions();
            serializerOptions.WriteIndented = true;
            using (var appsettingWriteStream = File.Open(appsettingStreamFile, FileMode.Create))
                JsonSerializer.Serialize(appsettingWriteStream, appSettings, serializerOptions);
        }

        private static string GetAppSettingsFile()
        {
            return Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "appsettings.json");
        }
    }
}
