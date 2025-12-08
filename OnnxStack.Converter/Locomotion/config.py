vae_sample_size = 512
unet_sample_size = 64
cross_attention_dim = 768
context_size = 16
#motion_adapter_name = "guoyww/animatediff-motion-adapter-v1-5-3"
motion_adapter_name ="https://huggingface.co/ByteDance/AnimateDiff-Lightning/blob/main/animatediff_lightning_8step_diffusers.safetensors"
lora_adapters = [
    #"guoyww/animatediff-motion-lora-v1-5-3",
    #"guoyww/animatediff-motion-lora-zoom-out",
    #"guoyww/animatediff-motion-lora-zoom-in",
    #"guoyww/animatediff-motion-lora-pan-left",
    #"guoyww/animatediff-motion-lora-pan-right",
    #"guoyww/animatediff-motion-lora-tilt-down",
    #"guoyww/animatediff-motion-lora-tilt-up",
    #"guoyww/animatediff-motion-lora-rolling-clockwise",
    #"guoyww/animatediff-motion-lora-rolling-anticlockwise",
]
lora_adapter_scales = [
    #0.7,
    #0.4
]