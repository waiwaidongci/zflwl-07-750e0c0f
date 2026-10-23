from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path


ENTRY = os.environ.get("TIDE_SENSOR", "/app/recover_tides.py")
PERIODS = (12.4206, 12.0, 25.8193)


def _run(payload: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, ENTRY, str(payload), str(output)],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _station(index: int) -> dict[str, object]:
    harmonics = [
        {
            "period_hours": period,
            "amplitude": 0.72 + 0.11 * index + 0.13 * constituent,
            "phase_deg": 17.0 * index + 29.0 * constituent - 40.0,
        }
        for constituent, period in enumerate(PERIODS)
    ]
    return {"id": f"TIDE-{index:02d}AB", "harmonics": harmonics}


def _payload(count: int = 12) -> tuple[dict[str, object], dict[str, str], dict[str, dict[str, float]]]:
    stations = [_station(index) for index in range(count)]
    order = [(index * 7 + 3) % count for index in range(count)]
    expected_map: dict[str, str] = {}
    expected_calibration: dict[str, dict[str, float]] = {}
    records: list[dict[str, object]] = []
    times = [round(index * 1.0, 6) for index in range(144)]
    for record_index, station_index in enumerate(order):
        gain = 0.91 + 0.013 * record_index
        lag_minutes = -36.0 + 5.5 * record_index
        lag_hours = lag_minutes / 60.0
        values = []
        for time in times:
            tide = sum(
                harmonic["amplitude"]
                * math.cos(
                    math.tau * (time - lag_hours) / harmonic["period_hours"]
                    - math.radians(harmonic["phase_deg"])
                )
                for harmonic in stations[station_index]["harmonics"]
            )
            values.append(0.27 + 0.004 * time + gain * tide)
        record_id = f"GAUGE-{record_index:02d}XY"
        records.append(
            {"id": record_id, "times_hours": times, "values": values}
        )
        expected_map[record_id] = stations[station_index]["id"]
        expected_calibration[record_id] = {
            "gain": round(gain, 6),
            "lag_minutes": round(lag_minutes, 6),
        }
    return (
        {"schema": "tidal-cal-v1", "stations": stations, "records": records},
        expected_map,
        expected_calibration,
    )


def test_constituent_lag_inversion_recovers_four_station_map(tmp_path: Path) -> None:
    """题面：
    「`python3 /app/recover_tides.py <input.json> <output.json>` 调用。成功时不向标准输出或」
    """

    payload, expected, _ = _payload(4)
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert result["status"] == "ok"
    assert result["station_map"] == expected, "CASE-TIDE-01 CASE-OUT-01 CASE-STATION-01 CASE-OUTPUT-01"


def test_twelve_gauge_assignment_is_globally_consistent(tmp_path: Path) -> None:
    """题面：
    「`records` 与台站数量相同，记录号匹配 `GAUGE-[A-Z0-9]{4}`，同批不得重号。每条记录」
    """

    payload, expected, _ = _payload(12)
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["station_map"] == expected, "CASE-STATION-02 CASE-RECORD-01 CASE-MATCH-01 CASE-OUTPUT-03 CASE-INPUT-03"


def test_shared_gain_and_lag_are_recovered_from_spectrum(tmp_path: Path) -> None:
    """题面：
    「每项只含 `gain` 和 `lag_minutes`，均保留六位小数。」
    """

    payload, _, expected = _payload(8)
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    observed = result["calibration"]
    assert completed.returncode == 0
    assert set(observed) == set(expected)
    assert abs(observed["GAUGE-00XY"]["gain"] - expected["GAUGE-00XY"]["gain"]) < 1e-5
    assert abs(observed["GAUGE-07XY"]["lag_minutes"] - expected["GAUGE-07XY"]["lag_minutes"]) < 1e-5, "CASE-TIDE-02 CASE-HARMONIC-01 CASE-OUTPUT-02 CASE-HARMONIC-03"


def test_hourly_sampling_boundary_is_accepted(tmp_path: Path) -> None:
    """题面：
    「只能有 `id`、`times_hours`、`values`。采样为 144 点，从 0 小时开始，每小时取一个点；」
    """

    payload, expected, _ = _payload(5)
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["records"][0]["times_hours"]) == 144
    assert completed.returncode == 0
    assert result["station_map"] == expected, "CASE-SCHEMA-01 CASE-HARMONIC-02 CASE-RECORD-02 CASE-RECORD-03"


def test_duplicate_spectral_signatures_report_ambiguity(tmp_path: Path) -> None:
    """题面：
    「每一项只能有 `period_hours`、`amplitude`、`phase_deg`。周期必须与上述顺序一致，振幅」
    """

    payload, _, _ = _payload(4)
    payload["stations"][1]["harmonics"] = [
        dict(item) for item in payload["stations"][0]["harmonics"]
    ]
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert set(result) == {"status", "candidates"}
    assert result["status"] == "ambiguous"
    assert result["candidates"] >= 2, "CASE-STATION-03 CASE-MATCH-02"


def test_extra_top_level_key_is_refused(tmp_path: Path) -> None:
    """题面：
    「结构、类型或值域不合规时，不创建输出文件，并在标准错误写一行 `REFUSED: <原因>`，」
    """

    payload, _, _ = _payload(4)
    payload["operator"] = "night-shift"
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists(), "CASE-INPUT-01 CASE-INPUT-02 CASE-SAFE-01"
    assert not (tmp_path / "tide-report.json").exists()
    assert not (tmp_path / "tide-summary.json").exists()


def test_success_output_is_compact_sorted_and_newline_terminated(tmp_path: Path) -> None:
    """题面：
    「`TIDE-[A-Z0-9]{4}`，重复编号立即拒收。`harmonics` 固定按 12.4206、12.0、25.8193 小时排列，」
    """

    payload, _, _ = _payload(4)
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    raw = output.read_bytes()
    parsed = json.loads(raw)
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    assert completed.returncode == 0
    assert raw == canonical


def test_nonuniform_sample_grid_is_refused(tmp_path: Path) -> None:
    """题面：
    「退出码为 2。最终 JSON 不保留空格，键以字母升序写出，文件结尾放一个换行。判分不依赖」
    """

    payload, _, _ = _payload(4)
    payload["records"][0]["times_hours"][4] = 4.03
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()
    assert not (tmp_path / "tide-report.json").exists()
    assert not (tmp_path / "tide-summary.json").exists()


def test_zero_harmonic_amplitude_is_refused(tmp_path: Path) -> None:
    """题面：
    「正确匹配必须是总残差唯一的方案。存在并列最优时，返回 `status` 和 `candidates`；」
    """

    # CASE-HARMONIC-ZERO
    payload, _, _ = _payload(4)
    payload["stations"][0]["harmonics"][0]["amplitude"] = 0.0
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()
    assert not (tmp_path / "tide-report.json").exists()
    assert not (tmp_path / "tide-summary.json").exists()


def test_duplicate_station_ids_are_refused(tmp_path: Path) -> None:
    """题面：
    「都有 M2、S2、K1 三个分潮的已知振幅和相位；临时仪只记录了一个统一的增益和钟差，」
    """

    # CASE-STATION-DUP
    payload, _, _ = _payload(4)
    payload["stations"][1]["id"] = payload["stations"][0]["id"]
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()
    assert not (tmp_path / "tide-report.json").exists()
    assert not (tmp_path / "tide-summary.json").exists()


def test_record_count_mismatch_is_refused(tmp_path: Path) -> None:
    """题面：
    「标准错误写内容，只生成结果文件。」
    """

    # CASE-RECORD-MISMATCH
    payload, _, _ = _payload(4)
    payload["records"].pop()
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()
    assert not (tmp_path / "tide-report.json").exists()
    assert not (tmp_path / "tide-summary.json").exists()


def test_gate_links_every_record_to_one_station(tmp_path: Path) -> None:
    """题面：
    「因此波形仍可辨认。值班组不希望逐台猜档案，而是从整批记录一次性还原仪器归属和校正量。」
    """

    payload, expected, _ = _payload(6)
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert set(result["station_map"]) == set(expected)
    assert len(set(result["station_map"].values())) == len(expected)
    assert result["station_map"] == expected
