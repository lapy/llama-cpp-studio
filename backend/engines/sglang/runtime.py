"""Entrypoints for SGLang's separate language and diffusion servers."""


def sglang_server_argv(python_bin: str, runtime: str = "llm") -> list[str]:
    if runtime == "llm":
        return [python_bin, "-m", "sglang.launch_server"]
    if runtime == "diffusion":
        # cli.main exposes a console-script function but has no __main__ guard.
        # Use the environment's interpreter directly so relocated venvs do not
        # depend on an absolute shebang in bin/sglang.
        return [
            python_bin,
            "-c",
            "from sglang.cli.main import main; main()",
            "serve",
            "--model-type",
            "diffusion",
        ]
    raise ValueError(f"Unsupported SGLang runtime: {runtime}")
