from fractions import Fraction
from contextlib import nullcontext
from types import SimpleNamespace

import av
import numpy as np
import pytest

from app.pipeline.video import iter_video_frames, upright_bgr


@pytest.fixture
def vfr_video(tmp_path):
    path = tmp_path / "variable.mkv"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("ffv1", rate=25)
        stream.width, stream.height = 16, 8
        stream.pix_fmt = "bgr0"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        for i, pts in enumerate([0, 40, 120, 160]):
            frame = av.VideoFrame.from_ndarray(np.full((8, 16, 3), i * 40, np.uint8), format="bgr24")
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


def test_real_decoder_preserves_variable_presentation_times_and_pixels(vfr_video):
    frames = list(iter_video_frames(str(vfr_video)))
    assert [frame.timestamp_ms for frame in frames] == pytest.approx([0, 40, 120, 160])
    assert [frame.index for frame in frames] == [0, 1, 2, 3]
    assert [int(frame.image[0, 0, 0]) for frame in frames] == [0, 40, 80, 120]


def test_rotation_matches_display_matrix_counterclockwise_convention():
    source = np.array([[1, 2, 3], [4, 5, 6]])
    assert upright_bgr(source, 90).tolist() == [[3, 6], [2, 5], [1, 4]]
    with pytest.raises(ValueError, match="rotation"):
        upright_bgr(source, 17)


def test_corrupt_video_is_an_error_not_successful_empty_results(tmp_path):
    path = tmp_path / "corrupt.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises((ValueError, av.FFmpegError)):
        list(iter_video_frames(str(path)))


def test_real_rotated_video_with_nonzero_container_start(tmp_path):
    path = tmp_path / "rotated.mp4"
    pixels = np.zeros((16, 32, 3), np.uint8)
    pixels[:8, :16] = 200
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264rgb", rate=25)
        stream.width, stream.height, stream.pix_fmt = 32, 16, "rgb24"
        stream.options = {"crf": "0"}
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        stream.set_display_rotation(90)
        for pts in (1000, 1040):
            frame = av.VideoFrame.from_ndarray(pixels, format="bgr24")
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    with av.open(str(path)) as raw:
        assert raw.start_time == 1_000_000
        assert next(raw.decode(video=0)).rotation == 90
    decoded = list(iter_video_frames(str(path)))
    assert [f.timestamp_ms for f in decoded] == [0, 40]
    assert decoded[0].image.shape == (32, 16, 3)
    expected = np.zeros((32, 16, 3), np.uint8)
    expected[16:, :8] = 200
    np.testing.assert_array_equal(decoded[0].image, expected)


@pytest.mark.parametrize("timestamps", [[None], [0, 0], [40, 20]])
def test_decoder_rejects_missing_or_invalid_presentation_timestamps(monkeypatch, timestamps):
    stream = SimpleNamespace(start_time=0, time_base=Fraction(1, 1000))
    frames = [SimpleNamespace(pts=pts, time_base=Fraction(1, 1000), rotation=0,
                              to_ndarray=lambda **kwargs: np.zeros((2, 2, 3), np.uint8)) for pts in timestamps]
    container = SimpleNamespace(start_time=0, streams=SimpleNamespace(video=[stream]),
                                decode=lambda stream: iter(frames))
    monkeypatch.setattr(av, "open", lambda path: nullcontext(container))
    with pytest.raises(ValueError, match="timestamp"):
        list(iter_video_frames("invalid-timestamps"))


def test_decoder_rejects_container_without_video_stream(monkeypatch):
    container = SimpleNamespace(streams=SimpleNamespace(video=[]))
    monkeypatch.setattr(av, "open", lambda path: nullcontext(container))
    with pytest.raises(ValueError, match="No video stream"):
        list(iter_video_frames("audio-only"))
