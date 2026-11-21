"""The install-plan primitive vocabulary.

Extracted from ``schema.py`` so the schema module stays inside its size budget
and this vocabulary has a single importable home. ``schema`` re-exports the
name, so every ``from backend.engines.definitions.schema import
INSTALL_PRIMITIVES`` keeps working.
"""

from __future__ import annotations

from typing import FrozenSet

# Install-plan primitives the Phase 3 executor must implement (the plan's
# "install primitives must be shared and typed, not arbitrary shell").
# Recipes may only reference these names; raw argv is expressible solely via
# a step's ``run`` for genuinely host-level one-offs, never inside engines.
INSTALL_PRIMITIVES: FrozenSet[str] = frozenset(
    {
        "host.inspect",
        "prereqs.resolve_version",
        "prereqs.resolve_release_assets",
        "artifact.resolve_candidates",
        "archive.download_extract",
        "vcs.clone",
        "vcs.sync",
        "vcs.checkout",
        "vcs.apply_patch",
        "vcs.capture_identity",
        "vcs.workspace_checkout",
        "vcs.seal_checkout",
        "vcs.capture_commit",
        "vcs.prepare_build",
        # Reusable fork mechanisms: versioned, applicability-checked file
        # transforms and a declared builder environment.
        "vcs.apply_transform",
        "build.prepare_fork",
        "venv.create",
        "env.compose",
        "env.compose_ccache",
        "exec.run_logged",
        "toolchain.ensure",
        "artifact.build_wheels",
        "artifact.locate",
        "artifact.remove_bundle",
        "artifact.read_checksum",
        "archive.download",
        "archive.verify_sha256",
        "archive.verify_checksums_file",
        "archive.extract_tar",
        "artifact.materialize_cli",
        # Engine-owned prebuilt deltas: the resolved release plan carries the
        # published digest, and the unpacked binaries must prove they can run.
        "prebuilt.download",
        "prebuilt.preflight",
        "cgroup.select_concurrency",
        "pip.install_requirements",
        "pip.install_package",
        "pip.uninstall",
        "pip.build_wheel",
        "pip.install_wheels",
        "pip.capture_metadata",
        "fs.remove_tree",
        "fs.remove_file",
        "fs.assert_dir",
        "fs.ensure_dir",
        "fs.prune_files",
        "cmake.configure",
        "cmake.build",
        "smoke.test",
        "publish.install",
        "publish.record_update",
        "publish.forget",
        "publish.finalize",
    }
)

__all__ = ["INSTALL_PRIMITIVES"]
