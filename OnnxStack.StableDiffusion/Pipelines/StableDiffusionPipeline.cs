using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.Core.Video;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Diffusers.StableDiffusion;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.StableDiffusion.Tokenizers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Pipelines
{
    public class StableDiffusionPipeline : PipelineBase
    {
        protected readonly UNetConditionModel _controlNetUnet;
        protected readonly ITokenizer _tokenizer;
        protected readonly TextEncoderModel _textEncoder;

        protected UNetConditionModel _unet;
        protected AutoEncoderModel _vaeDecoder;
        protected AutoEncoderModel _vaeEncoder;

        protected List<DiffuserType> _supportedDiffusers;
        protected IReadOnlyList<SchedulerType> _supportedSchedulers;
        protected SchedulerOptions _defaultSchedulerOptions;

        protected sealed record BatchResultInternal(SchedulerOptions SchedulerOptions, DenseTensor<float> Result);

        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionPipeline" /> class.
        /// </summary>
        /// <param name="name">The pipeline name.</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="controlNetUnet">The control net unet.</param>
        /// <param name="diffusers">The diffusers.</param>
        /// <param name="defaultSchedulerOptions">The default scheduler options.</param>
        /// <param name="logger">The logger.</param>
        public StableDiffusionPipeline(string name, ITokenizer tokenizer, TextEncoderModel textEncoder, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, UNetConditionModel controlNetUnet, List<DiffuserType> diffusers = default, List<SchedulerType> schedulers = default, SchedulerOptions defaultSchedulerOptions = default, ILogger logger = default)
            : base(name, logger)
        {
            _unet = unet;
            _tokenizer = tokenizer;
            _textEncoder = textEncoder;
            _vaeDecoder = vaeDecoder;
            _vaeEncoder = vaeEncoder;
            _controlNetUnet = controlNetUnet;
            _supportedDiffusers = diffusers ?? new List<DiffuserType>
            {
                 DiffuserType.TextToImage,
                 DiffuserType.ImageToImage,
                 DiffuserType.ImageInpaintLegacy,
                 DiffuserType.ControlNet,
                 DiffuserType.ControlNetImage
            };
            if (_controlNetUnet is null)
                _supportedDiffusers.RemoveRange(new[] { DiffuserType.ControlNet, DiffuserType.ControlNetImage });

            _supportedSchedulers = schedulers ?? new List<SchedulerType>
            {
                SchedulerType.LMS,
                SchedulerType.Euler,
                SchedulerType.EulerAncestral,
                SchedulerType.DDPM,
                SchedulerType.DDIM,
                SchedulerType.KDPM2,
                SchedulerType.KDPM2Ancestral,
                SchedulerType.LCM
            };
            _defaultSchedulerOptions = defaultSchedulerOptions ?? new SchedulerOptions
            {
                InferenceSteps = 30,
                GuidanceScale = 7.5f,
                SchedulerType = SchedulerType.DDPM,
                TimestepSpacing = TimestepSpacingType.Trailing
            };
        }


        /// <summary>
        /// Gets the supported diffusers.
        /// </summary>
        public override IReadOnlyList<DiffuserType> SupportedDiffusers => _supportedDiffusers;

        /// <summary>
        /// Gets the supported schedulers.
        /// </summary>
        public override IReadOnlyList<SchedulerType> SupportedSchedulers => _supportedSchedulers;

        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override PipelineType PipelineType => PipelineType.StableDiffusion;

        /// <summary>
        /// Gets the default scheduler options.
        /// </summary>
        public override SchedulerOptions DefaultSchedulerOptions => _defaultSchedulerOptions;

        /// <summary>
        /// Gets the unet.
        /// </summary>
        public UNetConditionModel Unet => _unet;

        /// <summary>
        /// Gets the control net unet.
        /// </summary>
        public UNetConditionModel ControlNetUnet => _controlNetUnet;

        /// <summary>
        /// Gets the tokenizer.
        /// </summary>
        public ITokenizer Tokenizer => _tokenizer;

        /// <summary>
        /// Gets the text encoder.
        /// </summary>
        public TextEncoderModel TextEncoder => _textEncoder;

        /// <summary>
        /// Gets the vae decoder.
        /// </summary>
        public AutoEncoderModel VaeDecoder => _vaeDecoder;

        /// <summary>
        /// Gets the vae encoder.
        /// </summary>
        public AutoEncoderModel VaeEncoder => _vaeEncoder;


        /// <summary>
        /// Unloads the pipeline.
        /// </summary>
        /// <returns></returns>
        public override async Task UnloadAsync()
        {
            // TODO: deadlock on model dispose when no synchronization context exists(console app)
            // Task.Yield seems to force a context switch resolving any issues, revist this
            await Task.Yield();

            _unet?.Dispose();
            _controlNetUnet?.Dispose();
            _textEncoder?.Dispose();
            _vaeDecoder?.Dispose();
            _vaeEncoder?.Dispose();
        }


        /// <summary>
        /// Runs the pipeline.
        /// </summary>
        /// <param name="options"></param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async Task<DenseTensor<float>> RunAsync(GenerateOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            return await RunInternalAsync(options, progressCallback, cancellationToken);
        }


        /// <summary>
        /// Runs the pipeline batch.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async IAsyncEnumerable<BatchResult> RunBatchAsync(GenerateBatchOptions options, IProgress<DiffusionProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await foreach (var batchResult in RunInternalAsync(options, progressCallback, cancellationToken))
            {
                yield return new BatchResult(batchResult.SchedulerOptions, batchResult.Result);
            }
        }


        /// <summary>
        /// Runs the pipeline returning the result as an OnnxImage.
        /// </summary>
        /// <param name="options"></param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async Task<OnnxImage> GenerateAsync(GenerateOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            return new OnnxImage(await RunInternalAsync(options, progressCallback, cancellationToken));
        }


        /// <summary>
        /// Runs the batch pipeline returning the result as an OnnxImage.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async IAsyncEnumerable<BatchImageResult> GenerateBatchAsync(GenerateBatchOptions options, IProgress<DiffusionProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await foreach (var batchResult in RunInternalAsync(options, progressCallback, cancellationToken))
            {
                yield return new BatchImageResult(batchResult.SchedulerOptions, new OnnxImage(batchResult.Result));
            }
        }


        /// <summary>
        /// Runs the pipeline returning the result as an OnnxVideo.
        /// </summary>
        /// <param name="options"></param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async Task<OnnxVideo> GenerateVideoAsync(GenerateOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var schedulerOptions = GetSchedulerOptionsOrDefault(options.SchedulerOptions);
            var tensors = await RunVideoInternalAsync(options, progressCallback, cancellationToken).ToListAsync(cancellationToken: cancellationToken);
            progressCallback.Notify("Generating Video Result...");
            return new OnnxVideo(tensors, options.OutputFrameRate);
        }


        /// <summary>
        /// Runs the pipeline returning each frame an OnnxImage.
        /// </summary>
        /// <param name="options"></param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async IAsyncEnumerable<OnnxImage> GenerateVideoFramesAsync(GenerateOptions options, IProgress<DiffusionProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var schedulerOptions = GetSchedulerOptionsOrDefault(options.SchedulerOptions);
            await foreach (var tensor in RunVideoInternalAsync(options, progressCallback, cancellationToken))
            {
                yield return new OnnxImage(tensor);
            }
        }


        /// <summary>
        /// Runs the video stream pipeline returning each frame as an OnnxImage.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="videoFrames">The video frames.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async IAsyncEnumerable<OnnxImage> GenerateVideoStreamAsync(GenerateOptions options, IAsyncEnumerable<OnnxImage> videoFrames, IProgress<DiffusionProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            _logger?.LogInformation("Diffuser starting...");
            var schedulerOptions = GetSchedulerOptionsOrDefault(options.SchedulerOptions);
            _logger?.Log($"Model: {Name}, Pipeline: {PipelineType}, Diffuser: {options.Diffuser}, Scheduler: {schedulerOptions.SchedulerType}");

            await CheckPipelineState(options);

            // Process prompts
            var promptEmbeddings = await CreatePromptEmbedsAsync(options, cancellationToken);

            // Create Diffuser
            var diffuser = CreateDiffuser(options.Diffuser, options.ControlNet);

            // Diffuse
            await foreach (var videoFrame in videoFrames)
            {
                var frameOptions = options with
                {
                    SchedulerOptions = schedulerOptions,
                    InputImage = options.Diffuser == DiffuserType.ImageToImage || options.Diffuser == DiffuserType.ControlNetImage
                        ? videoFrame : default,
                    InputContolImage = options.Diffuser == DiffuserType.ControlNet || options.Diffuser == DiffuserType.ControlNetImage
                        ? videoFrame : default,
                };

                yield return new OnnxImage(await DiffuseImageAsync(diffuser, frameOptions, promptEmbeddings, progressCallback, cancellationToken));
            }

            _logger?.LogInformation($"Diffuser complete");
        }


        /// <summary>
        /// Runs the image pipeline
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        protected virtual async Task<DenseTensor<float>> RunInternalAsync(GenerateOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            progressCallback.Notify("Loading Pipeline...");
            _logger?.LogInformation("Diffuser starting...");
            var schedulerOptions = GetSchedulerOptionsOrDefault(options.SchedulerOptions);
            _logger?.Log($"Model: {Name}, Pipeline: {PipelineType}, Diffuser: {options.Diffuser}, Scheduler: {options.SchedulerOptions.SchedulerType}");
            _logger?.Log($"Size: {schedulerOptions.Width}x{schedulerOptions.Height}, Steps: {schedulerOptions.InferenceSteps}, Guidance: {schedulerOptions.GuidanceScale:F2}");

            await CheckPipelineState(options);

            // Process prompts
            var promptEmbeddings = await CreatePromptEmbedsAsync(options, cancellationToken);

            // Create Diffuser
            var diffuser = CreateDiffuser(options.Diffuser, options.ControlNet);

            // Diffuse
            var tensorResult = await DiffuseImageAsync(diffuser, options, promptEmbeddings, progressCallback, cancellationToken);
            _logger?.LogInformation($"Diffuser complete.");
            return tensorResult;
        }


        /// <summary>
        /// Runs the video pipeline
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        protected virtual async IAsyncEnumerable<DenseTensor<float>> RunVideoInternalAsync(GenerateOptions options, IProgress<DiffusionProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            progressCallback.Notify("Loading Pipeline...");
            _logger?.LogInformation("Diffuser starting...");
            var schedulerOptions = GetSchedulerOptionsOrDefault(options.SchedulerOptions);
            _logger?.Log($"Model: {Name}, Pipeline: {PipelineType}, Diffuser: {options.Diffuser}, Scheduler: {options.SchedulerOptions.SchedulerType}");
            _logger?.Log($"Size: {schedulerOptions.Width}x{schedulerOptions.Height}, Steps: {schedulerOptions.InferenceSteps}, Guidance: {schedulerOptions.GuidanceScale:F2}");

            await CheckPipelineState(options);

            // Process prompts
            var promptEmbeddings = await CreatePromptEmbedsAsync(options, cancellationToken);

            // Create Diffuser
            var diffuser = CreateDiffuser(options.Diffuser, options.ControlNet);

            // Diffuse
            var frameIndex = 1;
            var tensorResult = new List<DenseTensor<float>>();
            var frameSchedulerCallback = CreateBatchCallback(progressCallback, options.FrameCount, () => frameIndex);
            await foreach (var frameTensor in DiffuseVideoAsync(diffuser, options, promptEmbeddings, frameSchedulerCallback, cancellationToken))
            {
                frameIndex++;
                yield return frameTensor;
            }

            _logger?.LogInformation($"Diffuser complete.");
        }


        /// <summary>
        /// Runs the image batch pipeline
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        protected virtual async IAsyncEnumerable<BatchResultInternal> RunInternalAsync(GenerateBatchOptions options, IProgress<DiffusionProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            progressCallback.Notify("Loading Pipeline...");
            _logger?.LogInformation("Batch Diffuser starting...");
            var schedulerOptions = GetSchedulerOptionsOrDefault(options.SchedulerOptions);
            _logger?.Log($"Model: {Name}, Pipeline: {PipelineType}, Diffuser: {options.Diffuser}, Scheduler: {schedulerOptions.SchedulerType}");
            _logger?.Log($"Size: {schedulerOptions.Width}x{schedulerOptions.Height}, Steps: {schedulerOptions.InferenceSteps}, Guidance: {schedulerOptions.GuidanceScale:F2}");
            _logger?.Log($"BatchType: {options.BatchType}, ValueFrom: {options.ValueFrom}, ValueTo: {options.ValueTo}, Increment: {options.Increment}");

            await CheckPipelineState(options);

            // Process prompts
            var promptEmbeddings = await CreatePromptEmbedsAsync(options, cancellationToken);

            // Generate batch options
            var batchSchedulerOptions = BatchGenerator.GenerateBatch(this, options, schedulerOptions);

            // Create Diffuser
            var diffuser = CreateDiffuser(options.Diffuser, options.ControlNet);

            // Diffuse
            var batchIndex = 1;// TODO: Video batch callback shoud be (BatchIndex + FrameIndex), not (BatchIndex + StepIndex)
            var batchSchedulerCallback = CreateBatchCallback(progressCallback, batchSchedulerOptions.Count, () => batchIndex);
            foreach (var batchSchedulerOption in batchSchedulerOptions)
            {
                var batchItemOptions = options with { SchedulerOptions = batchSchedulerOption };
                var tensorResult = await DiffuseImageAsync(diffuser, batchItemOptions, promptEmbeddings, batchSchedulerCallback, cancellationToken);

                batchIndex++;
                yield return new BatchResultInternal(batchSchedulerOption, tensorResult);
            }
            _logger?.LogInformation($"Batch Diffuser complete.");
        }


        /// <summary>
        /// Gets the scheduler options or the default scheduler options
        /// </summary>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <returns></returns>
        private SchedulerOptions GetSchedulerOptionsOrDefault(SchedulerOptions schedulerOptions)
        {
            // Create random seed if none was set
            if (schedulerOptions == null)
                return _defaultSchedulerOptions with { Seed = Random.Shared.Next() };

            if (schedulerOptions.Seed <= 0)
                schedulerOptions.Seed = Random.Shared.Next();

            return schedulerOptions;
        }


        public void OverrideUnet(UNetConditionModel unet)
        {
            if (_unet != null)
                _unet.Dispose();

            _unet = unet;
        }


        /// <summary>
        /// Overrides the vae encoder with a custom implementation, Caller is responsible for model lifetime
        /// </summary>
        /// <param name="vaeEncoder">The vae encoder.</param>
        public void OverrideVaeEncoder(AutoEncoderModel vaeEncoder)
        {
            if (_vaeEncoder != null)
                _vaeEncoder.Dispose();

            _vaeEncoder = vaeEncoder;
        }


        /// <summary>
        /// Overrides the vae decoder with a custom implementation, Caller is responsible for model lifetime
        /// </summary>
        /// <param name="vaeDecoder">The vae decoder.</param>
        public void OverrideVaeDecoder(AutoEncoderModel vaeDecoder)
        {
            if (_vaeDecoder != null)
                _vaeDecoder.Dispose();

            _vaeDecoder = vaeDecoder;
        }


        /// <summary>
        /// Creates the diffuser.
        /// </summary>
        /// <param name="diffuserType">Type of the diffuser.</param>
        /// <param name="controlNetModel">The control net model.</param>
        /// <returns></returns>
        protected override IDiffuser CreateDiffuser(DiffuserType diffuserType, ControlNetModel controlNetModel)
        {
            if (_unet.ModelType == ModelType.Instruct)
            {
                return diffuserType switch
                {
                    DiffuserType.ImageToImage => new InstructDiffuser(_unet, _vaeDecoder, _vaeEncoder, _logger),
                    DiffuserType.ControlNetImage => new InstructControlNetDiffuser(controlNetModel, _controlNetUnet, _vaeDecoder, _vaeEncoder, _logger),
                    _ => throw new NotImplementedException()
                };
            }

            return diffuserType switch
            {
                DiffuserType.TextToImage => new TextDiffuser(_unet, _vaeDecoder, _vaeEncoder, _logger),
                DiffuserType.ImageToImage => new ImageDiffuser(_unet, _vaeDecoder, _vaeEncoder, _logger),
                DiffuserType.ImageInpaint => new InpaintDiffuser(_unet, _vaeDecoder, _vaeEncoder, _logger),
                DiffuserType.ImageInpaintLegacy => new InpaintLegacyDiffuser(_unet, _vaeDecoder, _vaeEncoder, _logger),
                DiffuserType.ControlNet => new ControlNetDiffuser(controlNetModel, _controlNetUnet, _vaeDecoder, _vaeEncoder, _logger),
                DiffuserType.ControlNetImage => new ControlNetImageDiffuser(controlNetModel, _controlNetUnet, _vaeDecoder, _vaeEncoder, _logger),
                _ => throw new NotImplementedException()
            };
        }


        /// <summary>
        /// Checks the state of the pipeline.
        /// </summary>
        /// <param name="options">The options.</param>
        protected virtual async Task CheckPipelineState(GenerateOptions options)
        {
            switch (options.Diffuser)
            {
                case DiffuserType.TextToImage:
                case DiffuserType.ImageToImage:
                case DiffuserType.ImageInpaint:
                case DiffuserType.ImageInpaintLegacy:
                case DiffuserType.TextToVideo:
                case DiffuserType.ImageToVideo:
                case DiffuserType.VideoToVideo:
                    if (_controlNetUnet is not null)
                        await _controlNetUnet.UnloadAsync();
                    break;
                case DiffuserType.ControlNet:
                case DiffuserType.ControlNetImage:
                case DiffuserType.ControlNetVideo:
                    if (_unet is not null)
                        await _unet.UnloadAsync();
                    break;
                default:
                    break;
            }

            if (options.IsLowMemoryTextEncoderEnabled && _textEncoder?.Session is not null)
                await _textEncoder.UnloadAsync();
            if (options.IsLowMemoryEncoderEnabled && _vaeEncoder?.Session is not null)
                await _vaeEncoder.UnloadAsync();
            if (options.IsLowMemoryComputeEnabled && _unet?.Session is not null)
                await _unet.UnloadAsync();
            if (options.IsLowMemoryComputeEnabled && _controlNetUnet?.Session is not null)
                await _controlNetUnet.UnloadAsync();
            if (options.IsLowMemoryDecoderEnabled && _vaeDecoder?.Session is not null)
                await _vaeDecoder.UnloadAsync();
        }


        /// <summary>
        /// Creates the prompt embeds.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns></returns>
        protected virtual async Task<PromptEmbeddingsResult> CreatePromptEmbedsAsync(GenerateOptions options, CancellationToken cancellationToken = default)
        {
            // Tokenize Prompt and NegativePrompt
            var timestamp = _logger?.LogBegin();
            var promptTokens = await DecodePromptTextAsync(options.Prompt, cancellationToken);
            var negativePromptTokens = await DecodePromptTextAsync(options.NegativePrompt, cancellationToken);
            var maxPromptTokenCount = Math.Max(promptTokens.InputIds.Length, negativePromptTokens.InputIds.Length);
            _logger?.LogEnd(LogLevel.Debug, $"Tokenizer", timestamp);

            // Generate embeds for tokens
            timestamp = _logger?.LogBegin();
            var promptEmbeddings = await GeneratePromptEmbedsAsync(promptTokens, maxPromptTokenCount, cancellationToken);
            var negativePromptEmbeddings = await GeneratePromptEmbedsAsync(negativePromptTokens, maxPromptTokenCount, cancellationToken);
            _logger?.LogEnd(LogLevel.Debug, $"TextEncoder", timestamp);

            if (options.IsLowMemoryTextEncoderEnabled)
            {
                await _textEncoder.UnloadAsync();
            }

            return new PromptEmbeddingsResult(promptEmbeddings.PromptEmbeds, promptEmbeddings.PooledPromptEmbeds, negativePromptEmbeddings.PromptEmbeds, negativePromptEmbeddings.PooledPromptEmbeds);
        }


        /// <summary>
        /// Decodes the prompt text as tokens.
        /// </summary>
        /// <param name="inputText">The input text.</param>
        /// <returns></returns>
        protected virtual async Task<TokenizerResult> DecodePromptTextAsync(string inputText, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(inputText))
                return new TokenizerResult(Array.Empty<long>(), Array.Empty<long>());

            return await _tokenizer.EncodeAsync(inputText);
        }


        /// <summary>
        /// Encodes the prompt tokens.
        /// </summary>
        /// <param name="tokenizedInput">The tokenized input.</param>
        /// <returns></returns>
        protected virtual async Task<EncoderResult> EncodePromptTokensAsync(TokenizerResult tokenizedInput, CancellationToken cancellationToken = default)
        {
            var metadata = await _textEncoder.LoadAsync(cancellationToken: cancellationToken);
            var inputTensor = new DenseTensor<int>(tokenizedInput.InputIds.ToIntSafe(), new[] { 1, tokenizedInput.InputIds.Length });
            using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
            {
                inferenceParameters.AddInputTensor(inputTensor);
                inferenceParameters.AddOutputBuffer(new[] { 1, tokenizedInput.InputIds.Length, _tokenizer.TokenizerLength });
                inferenceParameters.AddOutputBuffer(new int[] { 1, _tokenizer.TokenizerLength });

                var results = await _textEncoder.RunInferenceAsync(inferenceParameters);
                using (var promptEmbeds = results.First())
                using (var promptEmbedsPooled = results.Last())
                {
                    return new EncoderResult(promptEmbeds.ToDenseTensor(), promptEmbedsPooled.ToDenseTensor());
                }
            }
        }


        /// <summary>
        /// Generates the prompt embeds.
        /// </summary>
        /// <param name="inputTokens">The input tokens.</param>
        /// <param name="minimumLength">The minimum length.</param>
        /// <returns></returns>
        protected async Task<EncoderResult> GeneratePromptEmbedsAsync(TokenizerResult inputTokens, int minimumLength, CancellationToken cancellationToken = default)
        {
            // If less than minimumLength pad with blank tokens
            if (inputTokens.InputIds.Length < minimumLength)
            {
                inputTokens.InputIds = PadWithBlankTokens(inputTokens.InputIds, minimumLength, _tokenizer.PadTokenId).ToArray();
                inputTokens.AttentionMask = PadWithBlankTokens(inputTokens.AttentionMask, minimumLength, 0).ToArray();
            }

            // The CLIP tokenizer only supports 77 tokens, batch process in groups of 77 and concatenate1
            var tokenBatches = new List<long[]>();
            var attentionBatches = new List<long[]>();
            foreach (var tokenBatch in inputTokens.InputIds.Chunk(_tokenizer.TokenizerLimit))
                tokenBatches.Add(PadWithBlankTokens(tokenBatch, _tokenizer.TokenizerLimit, _tokenizer.PadTokenId).ToArray());
            foreach (var attentionBatch in inputTokens.AttentionMask.Chunk(_tokenizer.TokenizerLimit))
                attentionBatches.Add(PadWithBlankTokens(attentionBatch, _tokenizer.TokenizerLimit, 0).ToArray());

            var promptEmbeddings = new List<float>();
            var pooledPromptEmbeddings = new List<float>();
            for (int i = 0; i < tokenBatches.Count; i++)
            {
                var result = await EncodePromptTokensAsync(new TokenizerResult(tokenBatches[i], attentionBatches[i]), cancellationToken);
                promptEmbeddings.AddRange(result.PromptEmbeds);
                pooledPromptEmbeddings.AddRange(result.PooledPromptEmbeds);
            }

            var promptTensor = new DenseTensor<float>(promptEmbeddings.ToArray(), new[] { 1, promptEmbeddings.Count / _tokenizer.TokenizerLength, _tokenizer.TokenizerLength });
            var pooledTensor = new DenseTensor<float>(pooledPromptEmbeddings.ToArray(), new[] { 1, tokenBatches.Count, _tokenizer.TokenizerLength });
            return new EncoderResult(promptTensor, pooledTensor);
        }


        /// <summary>
        /// Pads a source sequence with blank tokens if its less that the required length.
        /// </summary>
        /// <param name="inputs">The inputs.</param>
        /// <param name="requiredLength">The the required length of the returned array.</param>
        /// <returns></returns>
        protected IEnumerable<long> PadWithBlankTokens(IEnumerable<long> inputs, int requiredLength, int padTokenId)
        {
            var count = inputs.Count();
            if (requiredLength > count)
                return inputs.Concat(Enumerable.Repeat((long)padTokenId, requiredLength - count));
            return inputs;
        }


        /// <summary>
        /// Creates the pipeline from a ModelSet configuration.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static new StableDiffusionPipeline CreatePipeline(StableDiffusionModelSet modelSet, ILogger logger = default)
        {
            var config = modelSet with { };
            var unet = new UNetConditionModel(config.UnetConfig.ApplyDefaults(config));
            var tokenizer = new ClipTokenizer(config.TokenizerConfig.ApplyDefaults(config));
            var textEncoder = new TextEncoderModel(config.TextEncoderConfig.ApplyDefaults(config));
            var vaeDecoder = new AutoEncoderModel(config.VaeDecoderConfig.ApplyDefaults(config));
            var vaeEncoder = new AutoEncoderModel(config.VaeEncoderConfig.ApplyDefaults(config));
            var controlnet = default(UNetConditionModel);
            if (config.ControlNetUnetConfig is not null)
                controlnet = new UNetConditionModel(config.ControlNetUnetConfig.ApplyDefaults(config));

            LogPipelineInfo(modelSet, logger);
            return new StableDiffusionPipeline(config.Name, tokenizer, textEncoder, unet, vaeDecoder, vaeEncoder, controlnet, config.Diffusers, config.Schedulers, config.SchedulerOptions, logger);
        }


        /// <summary>
        /// Creates the pipeline from a folder structure.
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static StableDiffusionPipeline CreatePipeline(OnnxExecutionProvider executionProvider, string modelFolder, ModelType modelType = ModelType.Base, ILogger logger = default)
        {
            return CreatePipeline(ModelFactory.CreateStableDiffusionModelSet(modelFolder, modelType, PipelineType.StableDiffusion).WithProvider(executionProvider), logger);
        }
    }

}
