from clef_finetuning.config import load_experiments, validate


def test_all_configs_valid():
    assert validate() == []


def test_experiment_matrix_complete():
    expected = {"e0a", "e0b", "e0c", "e1a", "e1b", "e2a", "e2b", "e3", "e4", "e5", "e6", "e7", "e8"}
    assert set(load_experiments()) == expected


def test_every_track_has_experiments():
    tracks = {e.track for e in load_experiments().values()}
    assert tracks == {"baseline", "trl_a", "trl_b", "trl_c", "clef_native"}
