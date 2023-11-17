using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;

namespace OnnxStack.Core.Model
{
    public class OnnxInferenceParameters
    {
        private OnnxValueCollection _inputs;
        private OnnxValueCollection _outputs;

        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxInferenceParameters"/> class.
        /// </summary>
        public OnnxInferenceParameters()
        {
            _inputs = new OnnxValueCollection();
            _outputs = new OnnxValueCollection();
        }


        /// <summary>
        /// Adds an input parameter.
        /// </summary>
        /// <param name="metaData">The meta data.</param>
        /// <param name="value">The value.</param>
        public void AddInput(OnnxNamedMetadata metaData, OrtValue value)
        {
            _inputs.Add(metaData, value);
        }


        /// <summary>
        /// Adds an output parameter with known output size.
        /// </summary>
        /// <param name="metaData">The meta data.</param>
        /// <param name="value">The value.</param>
        public void AddOutput(OnnxNamedMetadata metaData, OrtValue value)
        {
            _outputs.Add(metaData, value);
        }


        /// <summary>
        /// Adds an output parameter with unknown output size.
        /// </summary>
        /// <param name="metaData">The meta data.</param>
        public void AddOutput(OnnxNamedMetadata metaData)
        {
            _outputs.AddName(metaData);
        }


        /// <summary>
        /// Gets the input names.
        /// </summary>
        public IReadOnlyCollection<string> InputNames => _inputs.Names;


        /// <summary>
        /// Gets the output names.
        /// </summary>
        public IReadOnlyCollection<string> OutputNames => _outputs.Names;


        /// <summary>
        /// Gets the input values.
        /// </summary>
        public IReadOnlyCollection<OrtValue> InputValues => _inputs.Values;


        /// <summary>
        /// Gets the output values.
        /// </summary>
        public IReadOnlyCollection<OrtValue> OutputValues => _outputs.Values;


        /// <summary>
        /// Gets the input name values.
        /// </summary>
        public IReadOnlyDictionary<string, OrtValue> InputNameValues => _inputs.NameValues;


        /// <summary>
        /// Gets the output name values.
        /// </summary>
        public IReadOnlyDictionary<string, OrtValue> OutputNameValues => _outputs.NameValues;
    }
}
