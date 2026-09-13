from backend.git_https import git_argv, is_network_git_command


def test_git_argv_rewrites_github_ssh_to_https():
    argv = git_argv("submodule", "update", "--init", "--recursive")
    assert argv[:1] == ["git"]
    assert argv[1:5] == [
        "-c",
        "url.https://github.com/.insteadOf=git@github.com:",
        "-c",
        "url.https://github.com/.insteadOf=ssh://git@github.com/",
    ]
    assert argv[5:] == ["submodule", "update", "--init", "--recursive"]


def test_git_argv_clone_recursive_keeps_source_url():
    argv = git_argv("clone", "--recursive", "https://github.com/0xShug0/audio.cpp.git", "/tmp/src")
    assert "clone" in argv
    assert "--recursive" in argv
    assert argv[-2:] == ["https://github.com/0xShug0/audio.cpp.git", "/tmp/src"]
    assert any("insteadOf=git@github.com:" in part for part in argv)


def test_is_network_git_command_detects_clone_fetch_and_submodule_update():
    assert is_network_git_command(git_argv("clone", "--recursive", "https://example.test/repo.git"))
    assert is_network_git_command(["git", "fetch", "--prune", "origin", "main"])
    assert is_network_git_command(["git", "submodule", "update", "--init", "--recursive"])
    assert not is_network_git_command(["git", "checkout", "-B", "main", "FETCH_HEAD"])
    assert not is_network_git_command(["git", "rev-parse", "HEAD"])
    assert not is_network_git_command(["cmake", "--build", "."])
