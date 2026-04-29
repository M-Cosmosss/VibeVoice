# vibevoice/modular/__init__.py
# Intentionally empty: avoid eager-loading streaming inference path
# (which pulls in diffusers via vibevoice.schedule.dpm_solver).
# Import submodules explicitly when needed.
