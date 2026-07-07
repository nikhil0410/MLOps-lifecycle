import pandas as pd

from scripts.download_and_prepare import prepare


def test_prepare_creates_clean_csv_and_binary_target(tmp_path):
    raw_path = tmp_path / "raw.csv"
    out_path = tmp_path / "processed.csv"

    rows = [
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, "?", 6, 0],
        [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3, 2],
    ]
    pd.DataFrame(rows).to_csv(raw_path, header=False, index=False)

    prepare(in_path=str(raw_path), out_path=str(out_path))

    assert out_path.exists()
    processed = pd.read_csv(out_path)
    assert list(processed.columns)[-1] == "target"
    assert set(processed["target"].tolist()) == {0, 1}
    assert pd.isna(processed.loc[0, "ca"])


def test_prepare_accepts_non_standard_last_column_name(tmp_path):
    raw_path = tmp_path / "raw_other.csv"
    out_path = tmp_path / "processed_other.csv"

    rows = [
        [10, 20, 0],
        [11, 21, 4],
    ]
    pd.DataFrame(rows).to_csv(raw_path, header=False, index=False)

    prepare(in_path=str(raw_path), out_path=str(out_path))

    processed = pd.read_csv(out_path)
    assert "target" in processed.columns
