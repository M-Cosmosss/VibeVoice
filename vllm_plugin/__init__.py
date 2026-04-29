"""VibeVoice vLLM Plugin - Registers VibeVoice model for vLLM inference.

This plugin enables VibeVoice ASR models to be loaded and served through vLLM.
It registers the model architecture, configuration, tokenizer, and processor
with their respective registries.

The plugin is automatically loaded by vLLM via the 'vllm.general_plugins'
entry point defined in pyproject.toml.
"""

from vllm.model_executor.models import ModelRegistry
from transformers import AutoConfig, AutoTokenizer, Qwen2Tokenizer, AutoProcessor

try:
    from transformers import Qwen2AudioProcessor
except ImportError:
    Qwen2AudioProcessor = None  # Base image transformers may not ship qwen2_audio

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceASRTextTokenizerFast

from .model import VibeVoiceForCausalLM


def register_vibevoice():
    """Register VibeVoice model with vLLM and transformers.
    
    This function is called automatically by vLLM through the entry point
    mechanism. It registers:
    - VibeVoiceConfig with AutoConfig
    - VibeVoiceASRTextTokenizerFast with AutoTokenizer (for ASR)
    - Qwen2AudioProcessor with AutoProcessor
    - VibeVoiceForCausalLM with vLLM ModelRegistry
    """
    # Register the configuration class with transformers
    try:
        AutoConfig.register("vibevoice", VibeVoiceConfig)
    except ValueError:
        pass  # May already be registered by base image

    # Register the tokenizer with transformers.
    # IMPORTANT (ASR): Align with the PyTorch ASR path.
    # VibeVoiceASRTextTokenizerFast maps:
    #   speech_start_id -> <|object_ref_start|>
    #   speech_pad_id   -> <|box_start|>
    #   speech_end_id   -> <|object_ref_end|>
    # This significantly affects ASR quality even when requests succeed.
    try:
        AutoTokenizer.register(
            VibeVoiceConfig,
            slow_tokenizer_class=Qwen2Tokenizer,
            fast_tokenizer_class=VibeVoiceASRTextTokenizerFast,
        )
    except Exception:
        pass  # May already be registered

    # Register the processor with transformers
    if Qwen2AudioProcessor is not None:
        try:
            AutoProcessor.register(VibeVoiceConfig, processor_class=Qwen2AudioProcessor)
        except Exception:
            pass  # May already be registered

    # Register the model class with the architecture name "VibeVoice"
    # This name must match the "architectures" list in config.json
    for arch in ("VibeVoice", "VibeVoiceForASRTraining"):
        try:
            ModelRegistry.register_model(arch, VibeVoiceForCausalLM)
        except (ValueError, KeyError):
            pass  # May already be registered (base image plugin / re-entry)


# Note: This function is called via vllm.general_plugins entry point
# defined in pyproject.toml, ensuring it runs in all vLLM processes
