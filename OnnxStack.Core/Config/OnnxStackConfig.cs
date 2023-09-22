using OnnxStack.Common.Config;
using System;
using System.Text.Json.Serialization;

namespace OnnxStack.Core.Config
{
    public class OnnxStackConfig : IConfigSection
    {
        private int _seed;

        /// <summary>
        /// Gets or sets the device identifier.
        /// </summary>
        /// <value>
        /// The device identifier used by DirectML and CUDA.
        /// </value>
        public int DeviceId { get; set; }

        /// <summary>
        /// Gets or sets the height.
        /// </summary>
        /// <value>
        ///  The height of the image. Default is 512 and must be a multiple of 8.
        /// </value>
        public int Height { get; set; } = 512;

        /// <summary>
        /// Gets or sets the width.
        /// </summary>
        /// <value>
        /// The width of the image. Default is 512 and must be a multiple of 8.
        /// </value>
        public int Width { get; set; } = 512;

        /// <summary>
        /// Gets or sets the number inference steps.
        /// </summary>
        /// <value>
        /// The number of steps to run inference for. The more steps the longer it will take to run the inference loop but the image quality should improve.
        /// </value>
        public int NumInferenceSteps { get; set; } = 15;

        /// <summary>
        /// Gets or sets the guidance scale.
        /// </summary>
        /// <value>
        /// The scale for the classifier-free guidance. The higher the number the more it will try to look like the prompt but the image quality may suffer.
        /// </value>
        public double GuidanceScale { get; set; } = 7.5;

        /// <summary>
        /// Gets or sets the execution provider target.
        /// </summary>
        public ExecutionProvider ExecutionProviderTarget { get; set; } = ExecutionProvider.DirectML;

        /// <summary>
        /// Gets or sets the seed.
        /// </summary>
        /// <value>
        /// If value is set to 0 a random seed is returned each call.
        /// </value>
        public int Seed
        {
            get { return _seed == 0 ? Random.Shared.Next() : _seed; }
            set { _seed = value; }
        }

        [JsonIgnore]
        public string OnnxTokenizerPath { get; set; }

        public string OnnxUnetPath { get; set; }
        public string OnnxVaeDecoderPath { get; set; }
        public string OnnxTextEncoderPath { get; set; }

        public void Initialize()
        {
            OnnxTokenizerPath = "cliptokenizer.onnx";
        }
      

    }
}
