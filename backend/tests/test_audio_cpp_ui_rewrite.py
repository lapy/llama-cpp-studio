"""Rewrite audio.cpp WebUI HTML to run under a public path prefix."""

from backend.audio_cpp_ui_rewrite import (
    llama_swap_upstream_prefix,
    rewrite_audio_cpp_ui_html,
    studio_audio_ui_prefix,
)


def test_studio_audio_ui_prefix():
    assert (
        studio_audio_ui_prefix("audio-cpp-pocket_tts_english_q8_0")
        == "/audio-cpp-ui/audio-cpp-pocket_tts_english_q8_0"
    )


def test_llama_swap_upstream_prefix():
    assert (
        llama_swap_upstream_prefix("audio-cpp-pocket_tts_english_q8_0")
        == "/upstream/audio-cpp-pocket_tts_english_q8_0"
    )


def test_rewrite_sets_sveltekit_base_and_fetch_bridge():
    html = """<!doctype html><html><head>
			<script>
				{
					__sveltekit_i8x7h6 = {
						base: ""
					};
				}
			</script>
		</head><body></body></html>"""
    out = rewrite_audio_cpp_ui_html(html, "audio-cpp-pocket_tts_english_q8_0").decode(
        "utf-8"
    )
    assert '__sveltekit_i8x7h6 = { base: "/audio-cpp-ui/audio-cpp-pocket_tts_english_q8_0" }' in out
    assert 'var p="/audio-cpp-ui/audio-cpp-pocket_tts_english_q8_0"' in out
    assert out.index("<script>(function(){") < out.index("__sveltekit_i8x7h6 = { base:")


def test_rewrite_accepts_llama_swap_upstream_prefix():
    html = '<html><head><script>__sveltekit_abc123 = { base: "" };</script></head></html>'
    out = rewrite_audio_cpp_ui_html(
        html, prefix="/upstream/audio-cpp-pocket_tts_english_q8_0"
    ).decode("utf-8")
    assert 'base: "/upstream/audio-cpp-pocket_tts_english_q8_0"' in out
    assert 'var p="/upstream/audio-cpp-pocket_tts_english_q8_0"' in out
