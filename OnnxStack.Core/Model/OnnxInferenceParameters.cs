using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;

namespace OnnxStack.Core.Model
{
    public class OnnxInferenceParameters : IDisposable
    {
        private readonly RunOptions _runOptions;
        private readonly OnnxMetadata _metadata;
        private readonly OnnxValueCollection _inputs;
        private readonly OnnxValueCollection _outputs;

        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxInferenceParameters"/> class.
        /// </summary>
        public OnnxInferenceParameters(OnnxMetadata metadata)
        {
            _metadata = metadata;
            _runOptions = new RunOptions();
            _inputs = new OnnxValueCollection();
            _outputs = new OnnxValueCollection();
        }


        /// <summary>
        /// Adds an input parameter.
        /// </summary>
        /// <param name="metaData">The meta data.</param>
        /// <param name="value">The value.</param>
        public void AddInput(OrtValue value)
        {
            _inputs.Add(GetNextInputMetadata(), value);
        }


        /// <summary>
        /// Adds the input tensor.
        /// </summary>
        /// <param name="value">The value.</param>
        public void AddInputTensor(DenseTensor<float> value)
        {
            var metaData = GetNextInputMetadata();
            _inputs.Add(metaData, value.ToOrtValue(metaData));
        }


        /// <summary>
        /// Adds the input tensor.
        /// </summary>
        /// <param name="value">The value.</param>
        public void AddInputTensor(DenseTensor<string> value)
        {
            var metaData = GetNextInputMetadata();
            _inputs.Add(metaData, value.ToOrtValue(metaData));
        }


        /// <summary>
        /// Adds the input tensor.
        /// </summary>
        /// <param name="value">The value.</param>
        public void AddInputTensor(DenseTensor<int> value)
        {
            var metaData = GetNextInputMetadata();
            _inputs.Add(metaData, value.ToOrtValue(metaData));
        }


        /// <summary>
        /// Adds an output parameter with known output size.
        /// </summary>
        /// <param name="metaData">The meta data.</param>
        /// <param name="value">The value.</param>
        public void AddOutput(OrtValue value)
        {
            _outputs.Add(GetNextOutputMetadata(), value);
        }


        /// <summary>
        /// Adds the output buffer.
        /// </summary>
        /// <param name="bufferDimension">The buffer dimension.</param>
        public void AddOutputBuffer(ReadOnlySpan<int> bufferDimension)
        {
            var metadata = GetNextOutputMetadata();
            _outputs.Add(metadata, metadata.CreateOutputBuffer(bufferDimension));
        }


        /// <summary>
        /// Adds an output parameter with unknown output size.
        /// </summary>
        /// <param name="metaData">The meta data.</param>
        public void AddOutputBuffer()
        {
            _outputs.AddName(GetNextOutputMetadata());
        }


        /// <summary>
        /// Gets the run options.
        /// </summary>
        public RunOptions RunOptions => _runOptions;

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


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            _inputs?.Dispose();
            _outputs?.Dispose();
            _runOptions?.Dispose();
        }

        private OnnxNamedMetadata GetNextInputMetadata()
        {
            if (_inputs.Names.Count >= _metadata.Inputs.Count)
                throw new ArgumentOutOfRangeException($"Too Many Inputs - No Metadata found for input index {_inputs.Names.Count - 1}");
   
            return _metadata.Inputs[_inputs.Names.Count];
        }

        private OnnxNamedMetadata GetNextOutputMetadata()
        {
            if (_outputs.Names.Count >= _metadata.Outputs.Count)
                throw new ArgumentOutOfRangeException($"Too Many Outputs - No Metadata found for output index {_outputs.Names.Count}");

            return _metadata.Outputs[_outputs.Names.Count];
        }
    }
}
