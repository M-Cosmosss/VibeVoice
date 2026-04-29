# vibevoice/__init__.py
# Intentionally empty: avoid eager-loading streaming / diffusion modules
# that would drag in heavy deps (diffusers, etc.) which the vLLM ASR
# plugin doesn't need. Consumers should import submodules explicitly,
# e.g. `from vibevoice.modular.configuration_vibevoice import ...`.
