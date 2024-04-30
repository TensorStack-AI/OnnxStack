using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
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
        protected readonly UNetConditionModel _unet;
        protected readonly TokenizerModel _tokenizer;
        protected readonly TextEncoderModel _textEncoder;

        protected AutoEncoderModel _vaeDecoder;
        protected AutoEncoderModel _vaeEncoder;
        protected OnnxModelSession _controlNet;
        protected List<DiffuserType> _supportedDiffusers;
        protected IReadOnlyList<SchedulerType> _supportedSchedulers;
        protected SchedulerOptions _defaultSchedulerOptions;

        protected sealed record BatchResultInternal(SchedulerOptions SchedulerOptions, List<DenseTensor<float>> Result);

        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionPipeline"/> class.
        /// </summary>
        /// <param name="pipelineOptions">The pipeline options</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public StableDiffusionPipeline(PipelineOptions pipelineOptions, TokenizerModel tokenizer, TextEncoderModel textEncoder, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, List<DiffuserType> diffusers = default, SchedulerOptions defaultSchedulerOptions = default, ILogger logger = default) : base(pipelineOptions, logger)
        {
            _unet = unet;
            _tokenizer = tokenizer;
            _textEncoder = textEncoder;
            _vaeDecoder = vaeDecoder;
            _vaeEncoder = vaeEncoder;
            _supportedDiffusers = diffusers ?? new List<DiffuserType>
            {
                 DiffuserType.TextToImage,
                 DiffuserType.ImageToImage,
                 DiffuserType.ImageInpaintLegacy
            };
            _supportedSchedulers = new List<SchedulerType>
            {
                SchedulerType.LMS,
                SchedulerType.Euler,
                SchedulerType.EulerAncestral,
                SchedulerType.DDPM,
                SchedulerType.DDIM,
                SchedulerType.KDPM2
            };
            _defaultSchedulerOptions = defaultSchedulerOptions ?? new SchedulerOptions
            {
                InferenceSteps = 30,
                GuidanceScale = 7.5f,
                SchedulerType = SchedulerType.EulerAncestral
            };
        }


        /// <summary>
        /// Gets the name.
        /// </summary>
        public override string Name => _pipelineOptions.Name;


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
        public override DiffuserPipelineType PipelineType => DiffuserPipelineType.StableDiffusion;


        /// <summary>
        /// Gets the default scheduler options.
        /// </summary>
        public override SchedulerOptions DefaultSchedulerOptions => _defaultSchedulerOptions;


        /// <summary>
        /// Loads the pipeline.
        /// </summary>
        public override Task LoadAsync()
        {
            if (_pipelineOptions.MemoryMode == MemoryModeType.Minimum)
                return Task.CompletedTask;

            // Preload all models into VRAM
            return Task.WhenAll
            (
                _unet.LoadAsync(),
                _tokenizer.LoadAsync(),
                _textEncoder.LoadAsync(),
                _vaeDecoder.LoadAsync(),
                _vaeEncoder.LoadAsync()
            );
        }


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
            _tokenizer?.Dispose();
            _textEncoder?.Dispose();
            _vaeDecoder?.Dispose();
            _vaeEncoder?.Dispose();
        }


        /// <summary>
        /// Validates the inputs.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        public override void ValidateInputs(PromptOptions promptOptions, SchedulerOptions schedulerOptions)
        {

        }


        /// <summary>
        /// Runs the pipeline returning the tensor result.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async Task<DenseTensor<float>> RunAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var tensors = await RunInternalAsync(promptOptions, schedulerOptions, controlNet, progressCallback, cancellationToken);
            return tensors.Count == 1
                ? tensors.First() // ImageTensor
                : tensors.Join(); // VideoTensor
        }


        /// <summary>
        /// Runs the pipeline batch.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="batchOptions">The batch options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async IAsyncEnumerable<BatchResult> RunBatchAsync(BatchOptions batchOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await foreach (var batchResult in RunBatchInternalAsync(batchOptions, promptOptions, schedulerOptions, controlNet, progressCallback, cancellationToken))
            {
                var tensor = batchResult.Result.Count == 1
                    ? batchResult.Result.First() // ImageTensor
                    : batchResult.Result.Join(); // VideoTensor
                yield return new BatchResult(batchResult.SchedulerOptions, tensor);
            }
        }


        /// <summary>
        /// Runs the pipeline returning the result as an OnnxImage.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async Task<OnnxImage> GenerateImageAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var tensors = await RunInternalAsync(promptOptions, schedulerOptions, controlNet, progressCallback, cancellationToken);
            return new OnnxImage(tensors.First());
        }


        /// <summary>
        /// Runs the batch pipeline returning the result as an OnnxImage.
        /// </summary>
        /// <param name="batchOptions">The batch options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async IAsyncEnumerable<BatchImageResult> GenerateImageBatchAsync(BatchOptions batchOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await foreach (var batchResult in RunBatchInternalAsync(batchOptions, promptOptions, schedulerOptions, controlNet, progressCallback, cancellationToken))
            {
                yield return new BatchImageResult(batchResult.SchedulerOptions, new OnnxImage(batchResult.Result.First()));
            }
        }


        /// <summary>
        /// Runs the pipeline returning the result as an OnnxVideo.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async Task<OnnxVideo> GenerateVideoAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var tensors = await RunInternalAsync(promptOptions, schedulerOptions, controlNet, progressCallback, cancellationToken);
            var videoInfo = promptOptions.InputVideo.Info with
            {
                Width = schedulerOptions.Width,
                Height = schedulerOptions.Height
            };
            progressCallback?.Invoke(new DiffusionProgress("Generating Video Result..."));
            return new OnnxVideo(videoInfo, tensors);
        }


        /// <summary>
        /// Runs the batch pipeline returning the result as an OnnxVideo.
        /// </summary>
        /// <param name="batchOptions">The batch options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async IAsyncEnumerable<BatchVideoResult> GenerateVideoBatchAsync(BatchOptions batchOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await foreach (var batchResult in RunBatchInternalAsync(batchOptions, promptOptions, schedulerOptions, controlNet, progressCallback, cancellationToken))
            {
                var videoInfo = promptOptions.InputVideo.Info with
                {
                    Width = batchResult.SchedulerOptions.Width,
                    Height = batchResult.SchedulerOptions.Height
                };
                progressCallback?.Invoke(new DiffusionProgress("Generating Video Result..."));
                yield return new BatchVideoResult(batchResult.SchedulerOptions, new OnnxVideo(videoInfo, batchResult.Result));
            }
        }


        /// <summary>
        /// Runs the video stream pipeline returning each frame as an OnnxImage.
        /// </summary>
        /// <param name="videoFrames">The video frames.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async IAsyncEnumerable<OnnxImage> GenerateVideoStreamAsync(IAsyncEnumerable<OnnxImage> videoFrames, PromptOptions promptOptions, SchedulerOptions schedulerOptions = null, ControlNetModel controlNet = null, Action<DiffusionProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var diffuseTime = _logger?.LogBegin("Diffuser starting...");
            var options = GetSchedulerOptionsOrDefault(schedulerOptions);
            _logger?.Log($"Model: {Name}, Pipeline: {PipelineType}, Diffuser: {promptOptions.DiffuserType}, Scheduler: {options.SchedulerType}");

            // Check guidance
            var performGuidance = ShouldPerformGuidance(options);

            // Process prompts
            var promptEmbeddings = await CreatePromptEmbedsAsync(promptOptions, performGuidance);

            // Create Diffuser
            var diffuser = CreateDiffuser(promptOptions.DiffuserType, controlNet);

            // Diffuse
            await foreach (var videoFrame in videoFrames)
            {
                var frameOptions = promptOptions with
                {
                    InputImage = promptOptions.DiffuserType == DiffuserType.ImageToImage || promptOptions.DiffuserType == DiffuserType.ControlNetImage
                        ? videoFrame : default,
                    InputContolImage = promptOptions.DiffuserType == DiffuserType.ControlNet || promptOptions.DiffuserType == DiffuserType.ControlNetImage
                        ? videoFrame : default,
                };

                yield return new OnnxImage(await DiffuseImageAsync(diffuser, frameOptions, options, promptEmbeddings, performGuidance, progressCallback, cancellationToken));
            }

            _logger?.LogEnd($"Diffuser complete", diffuseTime);
        }


        /// <summary>
        /// Runs the pipeline
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        protected virtual async Task<List<DenseTensor<float>>> RunInternalAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var diffuseTime = _logger?.LogBegin("Diffuser starting...");
            var options = GetSchedulerOptionsOrDefault(schedulerOptions);
            _logger?.Log($"Model: {Name}, Pipeline: {PipelineType}, Diffuser: {promptOptions.DiffuserType}, Scheduler: {options.SchedulerType}");

            // Check guidance
            var performGuidance = ShouldPerformGuidance(options);

            // Process prompts
            var promptEmbeddings = await CreatePromptEmbedsAsync(promptOptions, performGuidance);

            // Create Diffuser
            var diffuser = CreateDiffuser(promptOptions.DiffuserType, controlNet);

            // Diffuse
            var tensorResult = new List<DenseTensor<float>>();
            if (promptOptions.HasInputVideo)
            {
                var frameIndex = 1;
                var frameSchedulerCallback = CreateBatchCallback(progressCallback, promptOptions.InputVideo.Frames.Count, () => frameIndex);
                await foreach (var frameTensor in DiffuseVideoAsync(diffuser, promptOptions, options, promptEmbeddings, performGuidance, frameSchedulerCallback, cancellationToken))
                {
                    frameIndex++;
                    tensorResult.Add(frameTensor);
                }
            }
            else
            {
                tensorResult.Add(await DiffuseImageAsync(diffuser, promptOptions, options, promptEmbeddings, performGuidance, progressCallback, cancellationToken));
            }

            _logger?.LogEnd($"Diffuser complete", diffuseTime);
            return tensorResult;
        }


        /// <summary>
        /// Runs the pipeline batch.
        /// </summary>
        /// <param name="batchOptions">The batch options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        protected virtual async IAsyncEnumerable<BatchResultInternal> RunBatchInternalAsync(BatchOptions batchOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var diffuseBatchTime = _logger?.LogBegin("Batch Diffuser starting...");
            var options = GetSchedulerOptionsOrDefault(schedulerOptions);
            _logger?.Log($"Model: {Name}, Pipeline: {PipelineType}, Diffuser: {promptOptions.DiffuserType}, Scheduler: {options.SchedulerType}");
            _logger?.Log($"BatchType: {batchOptions.BatchType}, ValueFrom: {batchOptions.ValueFrom}, ValueTo: {batchOptions.ValueTo}, Increment: {batchOptions.Increment}");

            // Check guidance
            var performGuidance = ShouldPerformGuidance(options);

            // Process prompts
            var promptEmbeddings = await CreatePromptEmbedsAsync(promptOptions, performGuidance);

            // Generate batch options
            var batchSchedulerOptions = BatchGenerator.GenerateBatch(this, batchOptions, options);

            // Create Diffuser
            var diffuser = CreateDiffuser(promptOptions.DiffuserType, controlNet);

            // Diffuse
            var batchIndex = 1;// TODO: Video batch callback shoud be (BatchIndex + FrameIndex), not (BatchIndex + StepIndex)
            var batchSchedulerCallback = CreateBatchCallback(progressCallback, batchSchedulerOptions.Count, () => batchIndex);
            foreach (var batchSchedulerOption in batchSchedulerOptions)
            {
                var tensorResult = new List<DenseTensor<float>>();
                if (promptOptions.HasInputVideo)
                {
                    await foreach (var frameTensor in DiffuseVideoAsync(diffuser, promptOptions, batchSchedulerOption, promptEmbeddings, performGuidance, batchSchedulerCallback, cancellationToken))
                    {
                        tensorResult.Add(frameTensor);
                    }
                }
                else
                {
                    tensorResult.Add(await DiffuseImageAsync(diffuser, promptOptions, batchSchedulerOption, promptEmbeddings, performGuidance, batchSchedulerCallback, cancellationToken));
                }

                batchIndex++;
                yield return new BatchResultInternal(batchSchedulerOption, tensorResult);
            }
            _logger?.LogEnd($"Batch Diffuser complete", diffuseBatchTime);
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
            return diffuserType switch
            {
                DiffuserType.TextToImage => new TextDiffuser(_unet, _vaeDecoder, _vaeEncoder, _pipelineOptions.MemoryMode, _logger),
                DiffuserType.ImageToImage => new ImageDiffuser(_unet, _vaeDecoder, _vaeEncoder, _pipelineOptions.MemoryMode, _logger),
                DiffuserType.ImageInpaint => new InpaintDiffuser(_unet, _vaeDecoder, _vaeEncoder, _pipelineOptions.MemoryMode, _logger),
                DiffuserType.ImageInpaintLegacy => new InpaintLegacyDiffuser(_unet, _vaeDecoder, _vaeEncoder, _pipelineOptions.MemoryMode, _logger),
                DiffuserType.ControlNet => new ControlNetDiffuser(controlNetModel, _unet, _vaeDecoder, _vaeEncoder, _pipelineOptions.MemoryMode, _logger),
                DiffuserType.ControlNetImage => new ControlNetImageDiffuser(controlNetModel, _unet, _vaeDecoder, _vaeEncoder, _pipelineOptions.MemoryMode, _logger),
                _ => throw new NotImplementedException()
            };
        }


        /// <summary>
        /// Creates the prompt embeds.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="isGuidanceEnabled">if set to <c>true</c> [is guidance enabled].</param>
        /// <returns></returns>
        protected virtual async Task<PromptEmbeddingsResult> CreatePromptEmbedsAsync(PromptOptions promptOptions, bool isGuidanceEnabled)
        {
            // Tokenize Prompt and NegativePrompt
            var promptTokens = await DecodePromptTextAsync(promptOptions.Prompt);
            var negativePromptTokens = await DecodePromptTextAsync(promptOptions.NegativePrompt);
            var maxPromptTokenCount = Math.Max(promptTokens.InputIds.Length, negativePromptTokens.InputIds.Length);

            // Generate embeds for tokens
            var promptEmbeddings = await GeneratePromptEmbedsAsync(promptTokens, maxPromptTokenCount);
            var negativePromptEmbeddings = await GeneratePromptEmbedsAsync(negativePromptTokens, maxPromptTokenCount);


            if (_pipelineOptions.MemoryMode == MemoryModeType.Minimum)
            {
                await _tokenizer.UnloadAsync();
                await _textEncoder.UnloadAsync();
            }

            if (isGuidanceEnabled)
                return new PromptEmbeddingsResult(
                    negativePromptEmbeddings.PromptEmbeds.Concatenate(promptEmbeddings.PromptEmbeds),
                    negativePromptEmbeddings.PooledPromptEmbeds.Concatenate(promptEmbeddings.PooledPromptEmbeds));

            return new PromptEmbeddingsResult(promptEmbeddings.PromptEmbeds, promptEmbeddings.PooledPromptEmbeds);
        }


        /// <summary>
        /// Decodes the prompt text as tokens.
        /// </summary>
        /// <param name="inputText">The input text.</param>
        /// <returns></returns>
        protected async Task<TokenizerResult> DecodePromptTextAsync(string inputText)
        {
            if (string.IsNullOrEmpty(inputText))
                return new TokenizerResult(Array.Empty<long>(), Array.Empty<long>());

            var metadata = await _tokenizer.GetMetadataAsync();
            var inputTensor = new DenseTensor<string>(new string[] { inputText }, new int[] { 1 });
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(inputTensor);
                inferenceParameters.AddOutputBuffer();
                inferenceParameters.AddOutputBuffer();
                using (var results = _tokenizer.RunInference(inferenceParameters))
                {
                    return new TokenizerResult(results[0].ToArray<long>(), results[1].ToArray<long>());
                }
            }
        }


        /// <summary>
        /// Encodes the prompt tokens.
        /// </summary>
        /// <param name="tokenizedInput">The tokenized input.</param>
        /// <returns></returns>
        protected async Task<EncoderResult> EncodePromptTokensAsync(TokenizerResult tokenizedInput)
        {
            var metadata = await _textEncoder.GetMetadataAsync();
            var inputTensor = new DenseTensor<int>(tokenizedInput.InputIds.ToInt(), new[] { 1, tokenizedInput.InputIds.Length });
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
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
        protected async Task<PromptEmbeddingsResult> GeneratePromptEmbedsAsync(TokenizerResult inputTokens, int minimumLength)
        {
            // If less than minimumLength pad with blank tokens
            if (inputTokens.InputIds.Length < minimumLength)
            {
                inputTokens.InputIds = PadWithBlankTokens(inputTokens.InputIds, minimumLength, _tokenizer.PadTokenId).ToArray();
                inputTokens.AttentionMask = PadWithBlankTokens(inputTokens.AttentionMask, minimumLength, 1).ToArray();
            }

            // The CLIP tokenizer only supports 77 tokens, batch process in groups of 77 and concatenate1
            var tokenBatches = new List<long[]>();
            var attentionBatches = new List<long[]>();
            foreach (var tokenBatch in inputTokens.InputIds.Batch(_tokenizer.TokenizerLimit))
                tokenBatches.Add(PadWithBlankTokens(tokenBatch, _tokenizer.TokenizerLimit, _tokenizer.PadTokenId).ToArray());
            foreach (var attentionBatch in inputTokens.AttentionMask.Batch(_tokenizer.TokenizerLimit))
                attentionBatches.Add(PadWithBlankTokens(attentionBatch, _tokenizer.TokenizerLimit, 1).ToArray());

            var promptEmbeddings = new List<float>();
            var pooledPromptEmbeddings = new List<float>();
            for (int i = 0; i < tokenBatches.Count; i++)
            {
                var result = await EncodePromptTokensAsync(new TokenizerResult(tokenBatches[i], attentionBatches[i]));
                promptEmbeddings.AddRange(result.PromptEmbeds);
                pooledPromptEmbeddings.AddRange(result.PooledPromptEmbeds);
            }

            var promptTensor = new DenseTensor<float>(promptEmbeddings.ToArray(), new[] { 1, promptEmbeddings.Count / _tokenizer.TokenizerLength, _tokenizer.TokenizerLength });
            var pooledTensor = new DenseTensor<float>(pooledPromptEmbeddings.ToArray(), new[] { 1,  tokenBatches.Count, _tokenizer.TokenizerLength });
            return new PromptEmbeddingsResult(promptTensor, pooledTensor);
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
            var unet = new UNetConditionModel(modelSet.UnetConfig.ApplyDefaults(modelSet));
            var tokenizer = new TokenizerModel(modelSet.TokenizerConfig.ApplyDefaults(modelSet));
            var textEncoder = new TextEncoderModel(modelSet.TextEncoderConfig.ApplyDefaults(modelSet));
            var vaeDecoder = new AutoEncoderModel(modelSet.VaeDecoderConfig.ApplyDefaults(modelSet));
            var vaeEncoder = new AutoEncoderModel(modelSet.VaeEncoderConfig.ApplyDefaults(modelSet));
            var pipelineOptions = new PipelineOptions(modelSet.Name, modelSet.MemoryMode);
            return new StableDiffusionPipeline(pipelineOptions, tokenizer, textEncoder, unet, vaeDecoder, vaeEncoder, modelSet.Diffusers, modelSet.SchedulerOptions, logger);
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
        public static StableDiffusionPipeline CreatePipeline(string modelFolder, ModelType modelType = ModelType.Base, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML, MemoryModeType memoryMode = MemoryModeType.Maximum, ILogger logger = default)
        {
            return CreatePipeline(ModelFactory.CreateModelSet(modelFolder, DiffuserPipelineType.StableDiffusion, modelType, deviceId, executionProvider, memoryMode), logger);
        }
    }
}
