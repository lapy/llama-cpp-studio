"""Studio's native inference proxy.

The package deliberately has no imports at module load time.  This keeps the
legacy llama-swap backend usable while the native backend is selected at
runtime from settings.yaml.
"""
