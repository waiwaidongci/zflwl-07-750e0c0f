#!/usr/bin/env python3
"""Reference solver for anonymous tide-gauge calibration."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any


PERIODS = (12.4206, 12.0, 25.8193)
TOP_KEYS = {"schema", "stations", "records"}
STATION_ID = re.compile(r"TIDE-[A-Z0-9]{4}\Z")
RECORD_ID = re.compile(r"GAUGE-[A-Z0-9]{4}\Z")


class Refused(ValueError):
    pass


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refused(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise Refused(f"{label} is outside its domain")
    return result


def parse_payload(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Refused("input JSON is unreadable") from exc
    if not isinstance(payload, dict) or set(payload) != TOP_KEYS:
        raise Refused("top-level keys are invalid")
    if payload.get("schema") != "tidal-cal-v1":
        raise Refused("schema is unsupported")
    raw_stations = payload.get("stations")
    raw_records = payload.get("records")
    if not isinstance(raw_stations, list) or not 4 <= len(raw_stations) <= 18:
        raise Refused("stations must contain 4 to 18 entries")
    if not isinstance(raw_records, list) or len(raw_records) != len(raw_stations):
        raise Refused("records must match the station count")

    stations: list[dict[str, Any]] = []
    station_ids: set[str] = set()
    for item in raw_stations:
        if not isinstance(item, dict) or set(item) != {"id", "harmonics"}:
            raise Refused("station object keys are invalid")
        station_id = item.get("id")
        if not isinstance(station_id, str) or STATION_ID.fullmatch(station_id) is None:
            raise Refused("station id is invalid")
        if station_id in station_ids:
            raise Refused("station ids must be unique")
        station_ids.add(station_id)
        harmonics = item.get("harmonics")
        if not isinstance(harmonics, list) or len(harmonics) != 3:
            raise Refused("each station needs three harmonics")
        parsed: list[dict[str, float]] = []
        for index, harmonic in enumerate(harmonics):
            if not isinstance(harmonic, dict) or set(harmonic) != {
                "period_hours",
                "amplitude",
                "phase_deg",
            }:
                raise Refused("harmonic keys are invalid")
            period = _finite(harmonic.get("period_hours"), "period_hours", positive=True)
            if abs(period - PERIODS[index]) > 1e-9:
                raise Refused("harmonic periods are out of order")
            parsed.append(
                {
                    "period_hours": period,
                    "amplitude": _finite(
                        harmonic.get("amplitude"), "amplitude", positive=True
                    ),
                    "phase_deg": _finite(harmonic.get("phase_deg"), "phase_deg"),
                }
            )
        stations.append({"id": station_id, "harmonics": parsed})

    records: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    for item in raw_records:
        if not isinstance(item, dict) or set(item) != {"id", "times_hours", "values"}:
            raise Refused("record object keys are invalid")
        record_id = item.get("id")
        if not isinstance(record_id, str) or RECORD_ID.fullmatch(record_id) is None:
            raise Refused("record id is invalid")
        if record_id in record_ids:
            raise Refused("record ids must be unique")
        record_ids.add(record_id)
        times = item.get("times_hours")
        values = item.get("values")
        if not isinstance(times, list) or len(times) != 144:
            raise Refused("record must contain 144 samples")
        if not isinstance(values, list) or len(values) != 144:
            raise Refused("record values must contain 144 samples")
        parsed_times = [_finite(value, "sample time") for value in times]
        expected_times = [round(index * 1.0, 6) for index in range(144)]
        if parsed_times != expected_times:
            raise Refused("sample times must be a uniform hourly grid")
        parsed_values = [_finite(value, "tide reading") for value in values]
        records.append(
            {"id": record_id, "times_hours": parsed_times, "values": parsed_values}
        )
    return stations, records


def solve_three(matrix: list[list[float]], vector: list[float]) -> list[float]:
    work = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) < 1e-15:
            return [float("nan")] * 3
        work[column], work[pivot] = work[pivot], work[column]
        divisor = work[column][column]
        for index in range(column, 4):
            work[column][index] /= divisor
        for row in range(3):
            if row == column:
                continue
            factor = work[row][column]
            for index in range(column, 4):
                work[row][index] -= factor * work[column][index]
    return [work[index][3] for index in range(3)]


def station_template(
    station: dict[str, Any], times: list[float], lag_minutes: float
) -> list[float]:
    lag_hours = lag_minutes / 60.0
    return [
        sum(
            harmonic["amplitude"]
            * math.cos(
                math.tau * (time - lag_hours) / harmonic["period_hours"]
                - math.radians(harmonic["phase_deg"])
            )
            for harmonic in station["harmonics"]
        )
        for time in times
    ]


def fit_at_lag(
    observed: list[float],
    station: dict[str, Any],
    times: list[float],
    lag_minutes: float,
) -> tuple[float, float]:
    template = station_template(station, times, lag_minutes)
    rows = [[1.0, time, template[index]] for index, time in enumerate(times)]
    normal = [[0.0] * 3 for _ in range(3)]
    rhs = [0.0] * 3
    for row, value in zip(rows, observed):
        for left in range(3):
            rhs[left] += row[left] * value
            for right in range(3):
                normal[left][right] += row[left] * row[right]
    coefficients = solve_three(normal, rhs)
    gain = coefficients[2]
    if not math.isfinite(gain) or gain <= 0.0:
        return float("inf"), float("nan")
    residual = sum(
        (
            observed[index]
            - coefficients[0]
            - coefficients[1] * times[index]
            - gain * template[index]
        )
        ** 2
        for index in range(len(observed))
    )
    scale = max(1e-15, math.fsum(value * value for value in observed))
    return residual / scale, gain


def pair_cost(
    observed: list[float], station: dict[str, Any], times: list[float]
) -> tuple[float, float, float]:
    candidates = [
        (fit_at_lag(observed, station, times, lag)[0], float(lag))
        for lag in range(-90, 91, 2)
    ]
    _, center = min(candidates)
    left = center - 2.0
    right = center + 2.0
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    first = right - ratio * (right - left)
    second = left + ratio * (right - left)
    first_cost = fit_at_lag(observed, station, times, first)[0]
    second_cost = fit_at_lag(observed, station, times, second)[0]
    for _ in range(44):
        if first_cost <= second_cost:
            right = second
            second = first
            second_cost = first_cost
            first = right - ratio * (right - left)
            first_cost = fit_at_lag(observed, station, times, first)[0]
        else:
            left = first
            first = second
            first_cost = second_cost
            second = left + ratio * (right - left)
            second_cost = fit_at_lag(observed, station, times, second)[0]
    lag_minutes = (left + right) / 2.0
    residual, gain = fit_at_lag(observed, station, times, lag_minutes)
    return residual, gain, lag_minutes


def hungarian(cost: list[list[float]]) -> list[int]:
    size = len(cost)
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for row in range(1, size + 1):
        p[0] = row
        column0 = 0
        minv = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1][column - 1] - u[row0] - v[column]
                if current < minv[column]:
                    minv[column] = current
                    way[column] = column0
                if minv[column] < delta:
                    delta = minv[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minv[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = [0] * size
    for column in range(1, size + 1):
        if p[column]:
            assignment[p[column] - 1] = column - 1
    return assignment


def assignment_cost(cost: list[list[float]], assignment: list[int]) -> float:
    return math.fsum(cost[row][column] for row, column in enumerate(assignment))


def has_alternative_best(cost: list[list[float]], assignment: list[int]) -> bool:
    """Return whether a different full assignment ties the selected optimum."""
    best = assignment_cost(cost, assignment)
    second_best = math.inf
    for row, assigned_column in enumerate(assignment):
        restricted = [values[:] for values in cost]
        restricted[row][assigned_column] = math.inf
        alternative = hungarian(restricted)
        alternative_cost = assignment_cost(restricted, alternative)
        second_best = min(second_best, alternative_cost)
    return math.isfinite(second_best) and second_best - best <= 1e-9


def recover(stations: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    times = records[0]["times_hours"]
    observed = [record["values"] for record in records]
    costs: list[list[float]] = []
    parameters: list[list[tuple[float, float]]] = []
    for values in observed:
        row_cost: list[float] = []
        row_parameters: list[tuple[float, float]] = []
        for station in stations:
            residual, gain, lag = pair_cost(values, station, times)
            row_cost.append(residual)
            row_parameters.append((gain, lag))
        costs.append(row_cost)
        parameters.append(row_parameters)

    assignment = hungarian(costs)
    if has_alternative_best(costs, assignment):
        return {"candidates": 2, "status": "ambiguous"}

    station_map: dict[str, str] = {}
    calibration: dict[str, dict[str, float]] = {}
    for record_index, station_index in enumerate(assignment):
        record_id = records[record_index]["id"]
        station_map[record_id] = stations[station_index]["id"]
        gain, lag = parameters[record_index][station_index]
        calibration[record_id] = {
            "gain": round(gain, 6),
            "lag_minutes": round(lag, 6),
        }
    return {
        "status": "ok",
        "station_map": dict(sorted(station_map.items())),
        "calibration": dict(sorted(calibration.items())),
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("REFUSED: expected input and output paths", file=sys.stderr)
        return 2
    try:
        stations, records = parse_payload(Path(sys.argv[1]))
        result = recover(stations, records)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    Path(sys.argv[2]).write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
