"""Database helpers for nanostar sequences and their melting curves.

Curve points use the same shape as :mod:`delta_g_estimator`::

    {"temperature_C": 20.0, "structure": "...", "mfe_kcal_per_mol": -4.2}
"""

import json
from collections.abc import Iterable, Mapping

from django.db import connection, transaction

from .delta_g_estimator import fit_free_energy_thermodynamics


SEQUENCE_TABLE = "nanostar_sequences"
CURVE_TABLE = "nanostar_temperature_curves"

SEQUENCE_FIELDS = (
    "arm1",
    "arm2",
    "arm3",
    "arm4",
    "middle",
    "linker",
    "A_Domain",
    "H",
    "S",
)


def create_nanostar_tables():
    """Create the nanostar and melting-curve tables if they do not exist."""
    with connection.cursor() as cursor:
        cursor.execute(
            f'''CREATE TABLE IF NOT EXISTS {CURVE_TABLE} (
                id BIGSERIAL PRIMARY KEY,
                curve JSONB NOT NULL
            )'''
        )
        cursor.execute(
            f'''CREATE TABLE IF NOT EXISTS {SEQUENCE_TABLE} (
                id BIGSERIAL PRIMARY KEY,
                arm1 TEXT NOT NULL,
                arm2 TEXT NOT NULL,
                arm3 TEXT NOT NULL,
                arm4 TEXT NOT NULL,
                middle TEXT NOT NULL,
                linker TEXT NOT NULL,
                "A_Domain" TEXT NOT NULL,
                "H" DOUBLE PRECISION NOT NULL,
                "S" DOUBLE PRECISION NOT NULL,
                curve_id BIGINT NOT NULL REFERENCES {CURVE_TABLE}(id)
                    ON DELETE CASCADE
            )'''
        )


@transaction.atomic
def wipe_nanostar_tables():
    """Delete every nanostar and curve row, restarting both IDs at 1."""
    create_nanostar_tables()
    with connection.cursor() as cursor:
        cursor.execute(
            f"TRUNCATE TABLE {SEQUENCE_TABLE}, {CURVE_TABLE} "
            "RESTART IDENTITY CASCADE"
        )


def _normalise_rows(rows):
    if isinstance(rows, Mapping):
        return [dict(rows)]
    if not isinstance(rows, Iterable):
        raise TypeError("rows must be a mapping or an iterable of mappings")
    result = [dict(row) for row in rows]
    if not result:
        raise ValueError("rows cannot be empty")
    return result


def _validate_curve(curve):
    if not isinstance(curve, (list, tuple)) or not curve:
        raise ValueError("each melting curve must be a non-empty list of points")
    required = {"temperature_C", "structure", "mfe_kcal_per_mol"}
    for index, point in enumerate(curve):
        if not isinstance(point, Mapping) or not required.issubset(point):
            raise ValueError(
                f"curve point {index} must contain {sorted(required)}"
            )
    return [dict(point) for point in curve]


@transaction.atomic
def append_nanostar_rows(rows, temperature_curve=None):
    """Append one or more nanostars and their melting curves.

    ``rows`` may be one mapping or an iterable of mappings. Supply
    ``temperature_curve`` to share one curve among all rows, or put a ``curve``
    value in each row. A row may instead contain an existing ``curve_id``.
    If ``H`` and ``S`` are absent, they are fitted from that row's curve using
    the same calculation as ``delta_g_estimator``.

    Returns the inserted nanostar IDs in input order.
    """
    create_nanostar_tables()
    normalised_rows = _normalise_rows(rows)
    shared_curve_id = None

    with connection.cursor() as cursor:
        if temperature_curve is not None:
            shared_curve = _validate_curve(temperature_curve)
            cursor.execute(
                f"INSERT INTO {CURVE_TABLE} (curve) VALUES (%s::jsonb) RETURNING id",
                [json.dumps(shared_curve)],
            )
            shared_curve_id = cursor.fetchone()[0]

        inserted_ids = []
        for row in normalised_rows:
            missing = [field for field in SEQUENCE_FIELDS[:7] if field not in row]
            if missing:
                raise ValueError(f"row is missing required fields: {missing}")

            row_curve = row.get("curve")
            curve_id = row.get("curve_id", shared_curve_id)
            curve_for_fit = None

            if row_curve is not None:
                curve_for_fit = _validate_curve(row_curve)
                cursor.execute(
                    f"INSERT INTO {CURVE_TABLE} (curve) "
                    "VALUES (%s::jsonb) RETURNING id",
                    [json.dumps(curve_for_fit)],
                )
                curve_id = cursor.fetchone()[0]
            elif temperature_curve is not None:
                curve_for_fit = shared_curve

            if curve_id is None:
                raise ValueError(
                    "each row needs a curve, curve_id, or shared temperature_curve"
                )

            H = row.get("H")
            S = row.get("S")
            if H is None or S is None:
                if curve_for_fit is None:
                    raise ValueError("H and S are required when only curve_id is given")
                fitted_H, fitted_S = fit_free_energy_thermodynamics(curve_for_fit)
                H = fitted_H if H is None else H
                S = fitted_S if S is None else S

            cursor.execute(
                f'''INSERT INTO {SEQUENCE_TABLE}
                    (arm1, arm2, arm3, arm4, middle, linker, "A_Domain", "H", "S", curve_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id''',
                [
                    row["arm1"], row["arm2"], row["arm3"], row["arm4"],
                    row["middle"], row["linker"], row["A_Domain"],
                    float(H), float(S), int(curve_id),
                ],
            )
            inserted_ids.append(cursor.fetchone()[0])

    return inserted_ids
