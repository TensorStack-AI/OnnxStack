using OnnxStack.Common.Config;
using System;
using System.Text.Json.Serialization;

namespace OnnxStack.Core.Config
{
    public class OnnxStackConfig : IConfigSection
    {
        /// <summary>
        /// Gets or sets the device identifier.
        /// </summary>
        /// <value>
        /// The device identifier used by DirectML and CUDA.
        /// </value>
        public int DeviceId { get; set; }

        /// <summary>
        /// Gets or sets the execution provider target.
        /// </summary>
        public ExecutionProvider ExecutionProviderTarget { get; set; } = ExecutionProvider.DirectML;

        [JsonIgnore]
        public string OnnxTokenizerPath { get; set; }
        public string OnnxUnetPath { get; set; }
        public string OnnxVaeDecoderPath { get; set; }
        public string OnnxTextEncoderPath { get; set; }
        public string OnnxSafetyModelPath { get; set; }
        public bool IsSafetyModelEnabled { get; set; }

        public void Initialize()
        {
            OnnxTokenizerPath = "cliptokenizer.onnx";
        }
    }
}
