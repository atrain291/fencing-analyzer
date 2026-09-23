from fractions import Fraction

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
