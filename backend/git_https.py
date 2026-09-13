"""Force GitHub remotes over HTTPS so clones work without ssh(1).

audio.cpp (and some other trees) advertise SSH submodule URLs. Studio
containers typically have git but no ssh binary or deploy key.
"""

from __future__ import annotations

from typing import List, Sequence

# git -c accepts "key=value". Multiple insteadOf entries cover both URL styles.
GITHUB_HTTPS_INSTEAD_OF = (
    "url.https://github.com/.insteadOf=git@github.com:",
    "url.https://github.com/.insteadOf=ssh://git@github.com/",
)


def git_argv(*args: str) -> List[str]:
    """``git`` plus HTTPS-for-SSH rewrites, then the subcommand."""
    cmd = ["git"]
    for rewrite in GITHUB_HTTPS_INSTEAD_OF:
        cmd.extend(["-c", rewrite])
    cmd.extend(args)
    return cmd


def is_network_git_command(argv: Sequence[str]) -> bool:
    """True when argv is a git command that may contact remotes."""
    parts = [str(item) for item in argv]
    if not parts or parts[0] != "git":
        return False
    # Skip leading -c key=value pairs.
    index = 1
    while index + 1 < len(parts) and parts[index] == "-c":
        index += 2
    if index >= len(parts):
        return False
    command = parts[index]
    if command in {"clone", "fetch", "pull", "push", "ls-remote"}:
        return True
    return command == "submodule" and "update" in parts[index:]
