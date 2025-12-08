using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.Core.Video;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using System;
using System.Collections.Generic;

namespace OnnxStack.StableDiffusion.Config
{
    public record GenerateOptions
    {
        public GenerateOptions() { }
        public GenerateOptions(GenerateOptions options)
        {
            Diffuser = options.Diffuser;
            Prompt = options.Prompt;
            NegativePrompt = options.NegativePrompt;
            Prompts = options.Prompts;
            SchedulerOptions = options.SchedulerOptions with { };
            ControlNet = options.ControlNet;
            InputImage = options.InputImage;
            InputImageMask = options.InputImageMask;
            InputContolImage = options.InputContolImage;
            ClipSkip = options.ClipSkip;
            OptimizationType = options.OptimizationType;

            InputVideo = options.InputVideo;
            InputContolVideo = options.InputContolVideo;
            InputFrameRate = options.InputFrameRate;
            OutputFrameRate = options.OutputFrameRate;

            IsFrameBlendEnabled = options.IsFrameBlendEnabled;
            FrameStrength = options.FrameStrength;
            PreviousFrameStrength = options.PreviousFrameStrength;
            FrameBlendingMode = options.FrameBlendingMode;

            FrameResample = options.FrameResample;
            FrameUpSample = options.FrameUpSample;
            FrameDownSample = options.FrameDownSample;


            MotionFrames = options.MotionFrames;
            MotionStrides = options.MotionStrides;
            MotionContextOverlap = options.MotionContextOverlap;
            MotionNoiseContext = options.MotionNoiseContext;

            IsLowMemoryComputeEnabled = options.IsLowMemoryComputeEnabled;
            IsLowMemoryEncoderEnabled = options.IsLowMemoryEncoderEnabled;
            IsLowMemoryDecoderEnabled = options.IsLowMemoryDecoderEnabled;
            IsLowMemoryTextEncoderEnabled = options.IsLowMemoryTextEncoderEnabled;

            IsAutoEncoderTileEnabled = options.IsAutoEncoderTileEnabled;
            AutoEncoderTileOverlap = options.AutoEncoderTileOverlap;
            AutoEncoderTileMode = options.AutoEncoderTileMode;
        }


        public DiffuserType Diffuser { get; set; }


        public string Prompt { get; set; }
        public string NegativePrompt { get; set; }
        public List<string> Prompts { get; set; } = new List<string>();

        public SchedulerOptions SchedulerOptions { get; set; }

        public ControlNetModel ControlNet { get; set; }


        public OnnxImage InputImage { get; set; }
        public OnnxImage InputImageMask { get; set; }
        public OnnxImage InputContolImage { get; set; }


        public int ClipSkip { get; set; }
        public OptimizationType OptimizationType { get; set; } = OptimizationType.None;

        public bool HasInputImage => InputImage?.HasImage ?? false;
        public bool HasInputImageMask => InputImageMask?.HasImage ?? false;
        public bool HasInputContolImage => InputContolImage?.HasImage ?? false;

        public float InputFrameRate { get; set; }
        public float OutputFrameRate { get; set; }
        public OnnxVideo InputVideo { get; set; }
        public OnnxVideo InputContolVideo { get; set; }
        public bool IsFrameBlendEnabled { get; set; } = false;
        public float FrameStrength { get; set; } = 0.9f;
        public float PreviousFrameStrength { get; set; } = 0.3f;
        public ImageBlendingMode FrameBlendingMode { get; set; } = ImageBlendingMode.Overlay;

        public bool HasInputVideo => InputVideo?.HasVideo ?? false;
        public bool HasInputContolVideo => InputContolVideo?.HasVideo ?? false;
        public int FrameCount => HasInputVideo ? InputVideo.FrameCount : HasInputContolVideo ? InputContolVideo.FrameCount :MotionFrames;
        public float FrameRate => HasInputVideo
            ? InputVideo.FrameRate
            : HasInputContolVideo
            ? InputContolVideo.FrameRate
            : InputFrameRate > 0 ? InputFrameRate : 0;

        public bool FrameResample { get; set; } = false;
        public int FrameUpSample { get; set; } = 2;
        public int FrameDownSample { get; set; } = 0;

        public bool IsAutoEncoderTileEnabled { get; set; }
        public int AutoEncoderTileOverlap { get; set; } = 8;
        public TileMode AutoEncoderTileMode { get; set; } = TileMode.ClipBlend;

        public bool IsLowMemoryComputeEnabled { get; set; }
        public bool IsLowMemoryEncoderEnabled { get; set; }
        public bool IsLowMemoryDecoderEnabled { get; set; }
        public bool IsLowMemoryTextEncoderEnabled { get; set; }


        
        public int MotionFrames{ get; set; } = 16;
        public int MotionStrides { get; set; } = 0;
        public int MotionContextOverlap { get; set; } = 3;
        public int MotionNoiseContext { get; set; } = 16;
    }
}
