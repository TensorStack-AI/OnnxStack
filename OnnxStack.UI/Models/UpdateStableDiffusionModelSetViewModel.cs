using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.UI.Views;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace OnnxStack.UI.Models
{
    public class UpdateStableDiffusionModelSetViewModel : INotifyPropertyChanged
    {
        private string _name;
        private int _deviceId;
        private int _interOpNumThreads;
        private int _intraOpNumThreads;
        private ExecutionMode _executionMode;
        private ExecutionProvider _executionProvider;
        private bool _enableTextToImage;
        private bool _enableImageToImage;
        private bool _enableImageInpaint;
        private bool _enableImageInpaintLegacy;
        private bool _enableControlNet;
        private bool _enableControlNetImage;
        private DiffuserPipelineType _pipelineType;

        private int _sampleSize;

        private ModelFileViewModel _unetModel;
        private ModelFileViewModel _vaeEncoderModel;
        private ModelFileViewModel _vaeDecoderModel;
        private ModelFileViewModel _textEncoder2Model;
        private ModelFileViewModel _textEncoderModel;
        private ModelFileViewModel _tokenizer2Model;
        private ModelFileViewModel _tokenizerModel;


        private int _padTokenId;
        private int _blankTokenId;
        private float _scaleFactor;
        private int _tokenizerLimit;
        private int _tokenizerLength;
        private int _tokenizer2Length;

        public string Name
        {
            get { return _name; }
            set { _name = value; NotifyPropertyChanged(); }
        }

        public int PadTokenId
        {
            get { return _padTokenId; }
            set { _padTokenId = value; NotifyPropertyChanged(); }
        }
        public int BlankTokenId
        {
            get { return _blankTokenId; }
            set { _blankTokenId = value; NotifyPropertyChanged(); }
        }

        public int SampleSize
        {
            get { return _sampleSize; }
            set { _sampleSize = value; NotifyPropertyChanged(); }
        }

        public float ScaleFactor
        {
            get { return _scaleFactor; }
            set { _scaleFactor = value; NotifyPropertyChanged(); }
        }

        public int TokenizerLimit
        {
            get { return _tokenizerLimit; }
            set { _tokenizerLimit = value; NotifyPropertyChanged(); }
        }


        public int Tokenizer2Length
        {
            get { return _tokenizer2Length; }
            set { _tokenizer2Length = value; NotifyPropertyChanged(); }
        }

        public int TokenizerLength
        {
            get { return _tokenizerLength; }
            set { _tokenizerLength = value; NotifyPropertyChanged(); }
        }

        public bool EnableTextToImage
        {
            get { return _enableTextToImage; }
            set { _enableTextToImage = value; NotifyPropertyChanged(); }
        }

        public bool EnableImageToImage
        {
            get { return _enableImageToImage; }
            set { _enableImageToImage = value; NotifyPropertyChanged(); }
        }

        public bool EnableImageInpaint
        {
            get { return _enableImageInpaint; }
            set
            { _enableImageInpaint = value; NotifyPropertyChanged(); }
        }

        public bool EnableImageInpaintLegacy
        {
            get { return _enableImageInpaintLegacy; }
            set { _enableImageInpaintLegacy = value; NotifyPropertyChanged(); }
        }

        public bool EnableControlNet
        {
            get { return _enableControlNet; }
            set { _enableControlNet = value; NotifyPropertyChanged(); }
        }

        public bool EnableControlNetImage
        {
            get { return _enableControlNetImage; }
            set { _enableControlNetImage = value; NotifyPropertyChanged(); }
        }

        public int DeviceId
        {
            get { return _deviceId; }
            set { _deviceId = value; NotifyPropertyChanged(); }
        }

        public int InterOpNumThreads
        {
            get { return _interOpNumThreads; }
            set { _interOpNumThreads = value; NotifyPropertyChanged(); }
        }

        public int IntraOpNumThreads
        {
            get { return _intraOpNumThreads; }
            set { _intraOpNumThreads = value; NotifyPropertyChanged(); }
        }

        public ExecutionMode ExecutionMode
        {
            get { return _executionMode; }
            set { _executionMode = value; NotifyPropertyChanged(); }
        }

        public ExecutionProvider ExecutionProvider
        {
            get { return _executionProvider; }
            set { _executionProvider = value; NotifyPropertyChanged(); }
        }

        public DiffuserPipelineType PipelineType
        {
            get { return _pipelineType; }
            set { _pipelineType = value; NotifyPropertyChanged(); }
        }

        private ModelType _modelType;

        public ModelType ModelType
        {
            get { return _modelType; }
            set { _modelType = value; NotifyPropertyChanged(); }
        }


        public ModelFileViewModel UnetModel
        {
            get { return _unetModel; }
            set { _unetModel = value; NotifyPropertyChanged(); }
        }

        public ModelFileViewModel TokenizerModel
        {
            get { return _tokenizerModel; }
            set { _tokenizerModel = value; NotifyPropertyChanged(); }
        }

        public ModelFileViewModel Tokenizer2Model
        {
            get { return _tokenizer2Model; }
            set { _tokenizer2Model = value; NotifyPropertyChanged(); }
        }

        public ModelFileViewModel TextEncoderModel
        {
            get { return _textEncoderModel; }
            set { _textEncoderModel = value; NotifyPropertyChanged(); }
        }

        public ModelFileViewModel TextEncoder2Model
        {
            get { return _textEncoder2Model; }
            set { _textEncoder2Model = value; NotifyPropertyChanged(); }
        }

        public ModelFileViewModel VaeDecoderModel
        {
            get { return _vaeDecoderModel; }
            set { _vaeDecoderModel = value; NotifyPropertyChanged(); }
        }

        public ModelFileViewModel VaeEncoderModel
        {
            get { return _vaeEncoderModel; }
            set { _vaeEncoderModel = value; NotifyPropertyChanged(); }
        }


        public IEnumerable<DiffuserType> GetDiffusers()
        {
            if (_enableTextToImage)
                yield return DiffuserType.TextToImage;
            if (_enableImageToImage)
                yield return DiffuserType.ImageToImage;
            if (_enableImageInpaint)
                yield return DiffuserType.ImageInpaint;
            if (_enableImageInpaintLegacy)
                yield return DiffuserType.ImageInpaintLegacy;
            if (_enableControlNet)
                yield return DiffuserType.ControlNet;
            if (_enableControlNetImage)
                yield return DiffuserType.ControlNetImage;
        }



        public static UpdateStableDiffusionModelSetViewModel FromModelSet(StableDiffusionModelSet modelset)
        {
            return new UpdateStableDiffusionModelSetViewModel
            {
                DeviceId = modelset.DeviceId,
                EnableImageInpaint = modelset.Diffusers.Contains(DiffuserType.ImageInpaint),
                EnableImageInpaintLegacy = modelset.Diffusers.Contains(DiffuserType.ImageInpaintLegacy),
                EnableImageToImage = modelset.Diffusers.Contains(DiffuserType.ImageToImage),
                EnableTextToImage = modelset.Diffusers.Contains(DiffuserType.TextToImage),
                EnableControlNet = modelset.Diffusers.Contains(DiffuserType.ControlNet),
                EnableControlNetImage = modelset.Diffusers.Contains(DiffuserType.ControlNetImage),
                ExecutionMode = modelset.ExecutionMode,
                ExecutionProvider = modelset.ExecutionProvider,
                InterOpNumThreads = modelset.InterOpNumThreads,
                IntraOpNumThreads = modelset.IntraOpNumThreads,

                Name = modelset.Name,
                PipelineType = modelset.PipelineType,
                SampleSize = modelset.SampleSize,

                BlankTokenId = modelset.TokenizerConfig.BlankTokenId,
                PadTokenId = modelset.TokenizerConfig.PadTokenId,
                TokenizerLimit = modelset.TokenizerConfig.TokenizerLimit,
                TokenizerLength = modelset.TokenizerConfig.TokenizerLength,
                Tokenizer2Length = modelset.Tokenizer2Config?.TokenizerLength ?? 1280,
                ModelType = modelset.UnetConfig.ModelType,
                ScaleFactor = modelset.VaeDecoderConfig.ScaleFactor,


                UnetModel = new ModelFileViewModel
                {
                    OnnxModelPath = modelset.UnetConfig.OnnxModelPath,

                    DeviceId = modelset.UnetConfig.DeviceId ?? modelset.DeviceId,
                    ExecutionMode = modelset.UnetConfig.ExecutionMode ?? modelset.ExecutionMode,
                    ExecutionProvider = modelset.UnetConfig.ExecutionProvider ?? modelset.ExecutionProvider,
                    InterOpNumThreads = modelset.UnetConfig.InterOpNumThreads ?? modelset.InterOpNumThreads,
                    IntraOpNumThreads = modelset.UnetConfig.IntraOpNumThreads ?? modelset.IntraOpNumThreads,
                    IsOverrideEnabled =
                               modelset.UnetConfig.DeviceId.HasValue
                            || modelset.UnetConfig.ExecutionMode.HasValue
                            || modelset.UnetConfig.ExecutionProvider.HasValue
                            || modelset.UnetConfig.IntraOpNumThreads.HasValue
                            || modelset.UnetConfig.InterOpNumThreads.HasValue
                },

                TokenizerModel = new ModelFileViewModel
                {
                    OnnxModelPath = modelset.TokenizerConfig.OnnxModelPath,

                    DeviceId = modelset.TokenizerConfig.DeviceId ?? modelset.DeviceId,
                    ExecutionMode = modelset.TokenizerConfig.ExecutionMode ?? modelset.ExecutionMode,
                    ExecutionProvider = modelset.TokenizerConfig.ExecutionProvider ?? modelset.ExecutionProvider,
                    InterOpNumThreads = modelset.TokenizerConfig.InterOpNumThreads ?? modelset.InterOpNumThreads,
                    IntraOpNumThreads = modelset.TokenizerConfig.IntraOpNumThreads ?? modelset.IntraOpNumThreads,
                    IsOverrideEnabled =
                               modelset.TokenizerConfig.DeviceId.HasValue
                            || modelset.TokenizerConfig.ExecutionMode.HasValue
                            || modelset.TokenizerConfig.ExecutionProvider.HasValue
                            || modelset.TokenizerConfig.IntraOpNumThreads.HasValue
                            || modelset.TokenizerConfig.InterOpNumThreads.HasValue
                },

                Tokenizer2Model = modelset.Tokenizer2Config is null ? default : new ModelFileViewModel
                {
                    OnnxModelPath = modelset.Tokenizer2Config.OnnxModelPath,

                    DeviceId = modelset.Tokenizer2Config.DeviceId ?? modelset.DeviceId,
                    ExecutionMode = modelset.Tokenizer2Config.ExecutionMode ?? modelset.ExecutionMode,
                    ExecutionProvider = modelset.Tokenizer2Config.ExecutionProvider ?? modelset.ExecutionProvider,
                    InterOpNumThreads = modelset.Tokenizer2Config.InterOpNumThreads ?? modelset.InterOpNumThreads,
                    IntraOpNumThreads = modelset.Tokenizer2Config.IntraOpNumThreads ?? modelset.IntraOpNumThreads,
                    IsOverrideEnabled =
                               modelset.Tokenizer2Config.DeviceId.HasValue
                            || modelset.Tokenizer2Config.ExecutionMode.HasValue
                            || modelset.Tokenizer2Config.ExecutionProvider.HasValue
                            || modelset.Tokenizer2Config.IntraOpNumThreads.HasValue
                            || modelset.Tokenizer2Config.InterOpNumThreads.HasValue
                },

                TextEncoderModel = new ModelFileViewModel
                {
                    OnnxModelPath = modelset.TextEncoderConfig.OnnxModelPath,

                    DeviceId = modelset.TextEncoderConfig.DeviceId ?? modelset.DeviceId,
                    ExecutionMode = modelset.TextEncoderConfig.ExecutionMode ?? modelset.ExecutionMode,
                    ExecutionProvider = modelset.TextEncoderConfig.ExecutionProvider ?? modelset.ExecutionProvider,
                    InterOpNumThreads = modelset.TextEncoderConfig.InterOpNumThreads ?? modelset.InterOpNumThreads,
                    IntraOpNumThreads = modelset.TextEncoderConfig.IntraOpNumThreads ?? modelset.IntraOpNumThreads,
                    IsOverrideEnabled =
                               modelset.TextEncoderConfig.DeviceId.HasValue
                            || modelset.TextEncoderConfig.ExecutionMode.HasValue
                            || modelset.TextEncoderConfig.ExecutionProvider.HasValue
                            || modelset.TextEncoderConfig.IntraOpNumThreads.HasValue
                            || modelset.TextEncoderConfig.InterOpNumThreads.HasValue
                },

                TextEncoder2Model = modelset.TextEncoder2Config is null ? default : new ModelFileViewModel
                {
                    OnnxModelPath = modelset.TextEncoder2Config.OnnxModelPath,

                    DeviceId = modelset.TextEncoder2Config.DeviceId ?? modelset.DeviceId,
                    ExecutionMode = modelset.TextEncoder2Config.ExecutionMode ?? modelset.ExecutionMode,
                    ExecutionProvider = modelset.TextEncoder2Config.ExecutionProvider ?? modelset.ExecutionProvider,
                    InterOpNumThreads = modelset.TextEncoder2Config.InterOpNumThreads ?? modelset.InterOpNumThreads,
                    IntraOpNumThreads = modelset.TextEncoder2Config.IntraOpNumThreads ?? modelset.IntraOpNumThreads,
                    IsOverrideEnabled =
                               modelset.TextEncoder2Config.DeviceId.HasValue
                            || modelset.TextEncoder2Config.ExecutionMode.HasValue
                            || modelset.TextEncoder2Config.ExecutionProvider.HasValue
                            || modelset.TextEncoder2Config.IntraOpNumThreads.HasValue
                            || modelset.TextEncoder2Config.InterOpNumThreads.HasValue
                },

                VaeDecoderModel = new ModelFileViewModel
                {
                    OnnxModelPath = modelset.VaeDecoderConfig.OnnxModelPath,

                    DeviceId = modelset.VaeDecoderConfig.DeviceId ?? modelset.DeviceId,
                    ExecutionMode = modelset.VaeDecoderConfig.ExecutionMode ?? modelset.ExecutionMode,
                    ExecutionProvider = modelset.VaeDecoderConfig.ExecutionProvider ?? modelset.ExecutionProvider,
                    InterOpNumThreads = modelset.VaeDecoderConfig.InterOpNumThreads ?? modelset.InterOpNumThreads,
                    IntraOpNumThreads = modelset.VaeDecoderConfig.IntraOpNumThreads ?? modelset.IntraOpNumThreads,
                    IsOverrideEnabled =
                               modelset.VaeDecoderConfig.DeviceId.HasValue
                            || modelset.VaeDecoderConfig.ExecutionMode.HasValue
                            || modelset.VaeDecoderConfig.ExecutionProvider.HasValue
                            || modelset.VaeDecoderConfig.IntraOpNumThreads.HasValue
                            || modelset.VaeDecoderConfig.InterOpNumThreads.HasValue
                },

                VaeEncoderModel = new ModelFileViewModel
                {
                    OnnxModelPath = modelset.VaeEncoderConfig.OnnxModelPath,

                    DeviceId = modelset.VaeEncoderConfig.DeviceId ?? modelset.DeviceId,
                    ExecutionMode = modelset.VaeEncoderConfig.ExecutionMode ?? modelset.ExecutionMode,
                    ExecutionProvider = modelset.VaeEncoderConfig.ExecutionProvider ?? modelset.ExecutionProvider,
                    InterOpNumThreads = modelset.VaeEncoderConfig.InterOpNumThreads ?? modelset.InterOpNumThreads,
                    IntraOpNumThreads = modelset.VaeEncoderConfig.IntraOpNumThreads ?? modelset.IntraOpNumThreads,
                    IsOverrideEnabled =
                               modelset.VaeEncoderConfig.DeviceId.HasValue
                            || modelset.VaeEncoderConfig.ExecutionMode.HasValue
                            || modelset.VaeEncoderConfig.ExecutionProvider.HasValue
                            || modelset.VaeEncoderConfig.IntraOpNumThreads.HasValue
                            || modelset.VaeEncoderConfig.InterOpNumThreads.HasValue
                }

            };
        }

        public static StableDiffusionModelSet ToModelSet(UpdateStableDiffusionModelSetViewModel modelset)
        {
            return new StableDiffusionModelSet
            {
                IsEnabled = true,
                Name = modelset.Name,
                PipelineType = modelset.PipelineType,
                SampleSize = modelset.SampleSize,
                Diffusers = new List<DiffuserType>(modelset.GetDiffusers()),

                DeviceId = modelset.DeviceId,
                ExecutionMode = modelset.ExecutionMode,
                ExecutionProvider = modelset.ExecutionProvider,
                InterOpNumThreads = modelset.InterOpNumThreads,
                IntraOpNumThreads = modelset.IntraOpNumThreads,

                UnetConfig = new UNetConditionModelConfig
                {
                    ModelType = modelset.ModelType,
                    OnnxModelPath = modelset.UnetModel.OnnxModelPath,
                    DeviceId = modelset.UnetModel.IsOverrideEnabled && modelset.DeviceId != modelset.UnetModel.DeviceId ? modelset.UnetModel.DeviceId : default,
                    ExecutionMode = modelset.UnetModel.IsOverrideEnabled && modelset.ExecutionMode != modelset.UnetModel.ExecutionMode ? modelset.UnetModel.ExecutionMode : default,
                    ExecutionProvider = modelset.UnetModel.IsOverrideEnabled && modelset.ExecutionProvider != modelset.UnetModel.ExecutionProvider ? modelset.UnetModel.ExecutionProvider : default,
                    IntraOpNumThreads = modelset.UnetModel.IsOverrideEnabled && modelset.IntraOpNumThreads != modelset.UnetModel.IntraOpNumThreads ? modelset.UnetModel.IntraOpNumThreads : default,
                    InterOpNumThreads = modelset.UnetModel.IsOverrideEnabled && modelset.InterOpNumThreads != modelset.UnetModel.InterOpNumThreads ? modelset.UnetModel.InterOpNumThreads : default,
                },

                TokenizerConfig = new TokenizerModelConfig
                {
                    BlankTokenId = modelset.BlankTokenId,
                    PadTokenId = modelset.PadTokenId,
                    TokenizerLimit = modelset.TokenizerLimit,
                    TokenizerLength = modelset.TokenizerLength,
                    OnnxModelPath = modelset.TokenizerModel.OnnxModelPath,
                    DeviceId = modelset.TokenizerModel.IsOverrideEnabled && modelset.DeviceId != modelset.TokenizerModel.DeviceId ? modelset.TokenizerModel.DeviceId : default,
                    ExecutionMode = modelset.TokenizerModel.IsOverrideEnabled && modelset.ExecutionMode != modelset.TokenizerModel.ExecutionMode ? modelset.TokenizerModel.ExecutionMode : default,
                    ExecutionProvider = modelset.TokenizerModel.IsOverrideEnabled && modelset.ExecutionProvider != modelset.TokenizerModel.ExecutionProvider ? modelset.TokenizerModel.ExecutionProvider : default,
                    IntraOpNumThreads = modelset.TokenizerModel.IsOverrideEnabled && modelset.IntraOpNumThreads != modelset.TokenizerModel.IntraOpNumThreads ? modelset.TokenizerModel.IntraOpNumThreads : default,
                    InterOpNumThreads = modelset.TokenizerModel.IsOverrideEnabled && modelset.InterOpNumThreads != modelset.TokenizerModel.InterOpNumThreads ? modelset.TokenizerModel.InterOpNumThreads : default,
                },

                Tokenizer2Config = modelset.Tokenizer2Model is null ? default : new TokenizerModelConfig
                {
                    BlankTokenId = modelset.BlankTokenId,
                    PadTokenId = modelset.PadTokenId,
                    TokenizerLimit = modelset.TokenizerLimit,
                    TokenizerLength = modelset.Tokenizer2Length,
                    OnnxModelPath = modelset.Tokenizer2Model.OnnxModelPath,
                    DeviceId = modelset.Tokenizer2Model.IsOverrideEnabled && modelset.DeviceId != modelset.Tokenizer2Model.DeviceId ? modelset.Tokenizer2Model.DeviceId : default,
                    ExecutionMode = modelset.Tokenizer2Model.IsOverrideEnabled && modelset.ExecutionMode != modelset.Tokenizer2Model.ExecutionMode ? modelset.Tokenizer2Model.ExecutionMode : default,
                    ExecutionProvider = modelset.Tokenizer2Model.IsOverrideEnabled && modelset.ExecutionProvider != modelset.Tokenizer2Model.ExecutionProvider ? modelset.Tokenizer2Model.ExecutionProvider : default,
                    IntraOpNumThreads = modelset.Tokenizer2Model.IsOverrideEnabled && modelset.IntraOpNumThreads != modelset.Tokenizer2Model.IntraOpNumThreads ? modelset.Tokenizer2Model.IntraOpNumThreads : default,
                    InterOpNumThreads = modelset.Tokenizer2Model.IsOverrideEnabled && modelset.InterOpNumThreads != modelset.Tokenizer2Model.InterOpNumThreads ? modelset.Tokenizer2Model.InterOpNumThreads : default,
                },

                TextEncoderConfig = new TextEncoderModelConfig
                {
                    OnnxModelPath = modelset.TextEncoderModel.OnnxModelPath,
                    DeviceId = modelset.TextEncoderModel.IsOverrideEnabled && modelset.DeviceId != modelset.TextEncoderModel.DeviceId ? modelset.TextEncoderModel.DeviceId : default,
                    ExecutionMode = modelset.TextEncoderModel.IsOverrideEnabled && modelset.ExecutionMode != modelset.TextEncoderModel.ExecutionMode ? modelset.TextEncoderModel.ExecutionMode : default,
                    ExecutionProvider = modelset.TextEncoderModel.IsOverrideEnabled && modelset.ExecutionProvider != modelset.TextEncoderModel.ExecutionProvider ? modelset.TextEncoderModel.ExecutionProvider : default,
                    IntraOpNumThreads = modelset.TextEncoderModel.IsOverrideEnabled && modelset.IntraOpNumThreads != modelset.TextEncoderModel.IntraOpNumThreads ? modelset.TextEncoderModel.IntraOpNumThreads : default,
                    InterOpNumThreads = modelset.TextEncoderModel.IsOverrideEnabled && modelset.InterOpNumThreads != modelset.TextEncoderModel.InterOpNumThreads ? modelset.TextEncoderModel.InterOpNumThreads : default,
                },

                TextEncoder2Config = modelset.TextEncoder2Model is null ? default : new TextEncoderModelConfig
                {
                    OnnxModelPath = modelset.TextEncoder2Model.OnnxModelPath,
                    DeviceId = modelset.TextEncoder2Model.IsOverrideEnabled && modelset.DeviceId != modelset.TextEncoder2Model.DeviceId ? modelset.TextEncoder2Model.DeviceId : default,
                    ExecutionMode = modelset.TextEncoder2Model.IsOverrideEnabled && modelset.ExecutionMode != modelset.TextEncoder2Model.ExecutionMode ? modelset.TextEncoder2Model.ExecutionMode : default,
                    ExecutionProvider = modelset.TextEncoder2Model.IsOverrideEnabled && modelset.ExecutionProvider != modelset.TextEncoder2Model.ExecutionProvider ? modelset.TextEncoder2Model.ExecutionProvider : default,
                    IntraOpNumThreads = modelset.TextEncoder2Model.IsOverrideEnabled && modelset.IntraOpNumThreads != modelset.TextEncoder2Model.IntraOpNumThreads ? modelset.TextEncoder2Model.IntraOpNumThreads : default,
                    InterOpNumThreads = modelset.TextEncoder2Model.IsOverrideEnabled && modelset.InterOpNumThreads != modelset.TextEncoder2Model.InterOpNumThreads ? modelset.TextEncoder2Model.InterOpNumThreads : default,
                },

                VaeDecoderConfig = new AutoEncoderModelConfig
                {
                    ScaleFactor = modelset.ScaleFactor,
                    OnnxModelPath = modelset.VaeDecoderModel.OnnxModelPath,
                    DeviceId = modelset.VaeDecoderModel.IsOverrideEnabled && modelset.DeviceId != modelset.VaeDecoderModel.DeviceId ? modelset.VaeDecoderModel.DeviceId : default,
                    ExecutionMode = modelset.VaeDecoderModel.IsOverrideEnabled && modelset.ExecutionMode != modelset.VaeDecoderModel.ExecutionMode ? modelset.VaeDecoderModel.ExecutionMode : default,
                    ExecutionProvider = modelset.VaeDecoderModel.IsOverrideEnabled && modelset.ExecutionProvider != modelset.VaeDecoderModel.ExecutionProvider ? modelset.VaeDecoderModel.ExecutionProvider : default,
                    IntraOpNumThreads = modelset.VaeDecoderModel.IsOverrideEnabled && modelset.IntraOpNumThreads != modelset.VaeDecoderModel.IntraOpNumThreads ? modelset.VaeDecoderModel.IntraOpNumThreads : default,
                    InterOpNumThreads = modelset.VaeDecoderModel.IsOverrideEnabled && modelset.InterOpNumThreads != modelset.VaeDecoderModel.InterOpNumThreads ? modelset.VaeDecoderModel.InterOpNumThreads : default,
                },

                VaeEncoderConfig = new AutoEncoderModelConfig
                {
                    ScaleFactor = modelset.ScaleFactor,
                    OnnxModelPath = modelset.VaeEncoderModel.OnnxModelPath,
                    DeviceId = modelset.VaeEncoderModel.IsOverrideEnabled && modelset.DeviceId != modelset.VaeEncoderModel.DeviceId ? modelset.VaeEncoderModel.DeviceId : default,
                    ExecutionMode = modelset.VaeEncoderModel.IsOverrideEnabled && modelset.ExecutionMode != modelset.VaeEncoderModel.ExecutionMode ? modelset.VaeEncoderModel.ExecutionMode : default,
                    ExecutionProvider = modelset.VaeEncoderModel.IsOverrideEnabled && modelset.ExecutionProvider != modelset.VaeEncoderModel.ExecutionProvider ? modelset.VaeEncoderModel.ExecutionProvider : default,
                    IntraOpNumThreads = modelset.VaeEncoderModel.IsOverrideEnabled && modelset.IntraOpNumThreads != modelset.VaeEncoderModel.IntraOpNumThreads ? modelset.VaeEncoderModel.IntraOpNumThreads : default,
                    InterOpNumThreads = modelset.VaeEncoderModel.IsOverrideEnabled && modelset.InterOpNumThreads != modelset.VaeEncoderModel.InterOpNumThreads ? modelset.VaeEncoderModel.InterOpNumThreads : default,
                }

            };
        }



        #region INotifyPropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;
        public void NotifyPropertyChanged([CallerMemberName] string property = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(property));
        }

        #endregion
    }
}
