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
FULL_STRAND_TABLE = "nanostar_full_strands"
KINETIC_RATE_TABLE = "nanostar_kinetic_rates"

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

FULL_STRAND_FIELDS = (
    "full_arm1",
    "full_arm2",
    "full_arm3",
    "full_arm4",
    "upper_linker",
    "lower_linker",
)

KINETIC_RATE_FIELDS = ("k1", "k2", "k3", "k1m", "k2m", "k3m")


def create_nanostar_tables():
    """Create all nanostar data tables if needed."""
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
                kmeff DOUBLE PRECISION,
                keff DOUBLE PRECISION,
                curve_id BIGINT NOT NULL REFERENCES {CURVE_TABLE}(id)
                    ON DELETE CASCADE
            )'''
        )
        cursor.execute(
            f'''ALTER TABLE {SEQUENCE_TABLE}
                ADD COLUMN IF NOT EXISTS kmeff DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS keff DOUBLE PRECISION'''
        )
        cursor.execute(
            f'''CREATE TABLE IF NOT EXISTS {FULL_STRAND_TABLE} (
                id BIGSERIAL PRIMARY KEY,
                nanostar_id BIGINT NOT NULL UNIQUE
                    REFERENCES {SEQUENCE_TABLE}(id) ON DELETE CASCADE,
                full_arm1 TEXT NOT NULL,
                full_arm2 TEXT NOT NULL,
                full_arm3 TEXT NOT NULL,
                full_arm4 TEXT NOT NULL,
                upper_linker TEXT NOT NULL,
                lower_linker TEXT NOT NULL
            )'''
        )
        cursor.execute(
            f'''
                CREATE OR REPLACE FUNCTION delete_nanostar_for_full_strands()
                RETURNS TRIGGER
                LANGUAGE plpgsql
                AS $$
                BEGIN
                    IF pg_trigger_depth() = 1 THEN
                        DELETE FROM {SEQUENCE_TABLE}
                        WHERE id = OLD.nanostar_id;
                    END IF;
                    RETURN OLD;
                END;
                $$;
            '''
        )
        cursor.execute(
            f'''CREATE TABLE IF NOT EXISTS {KINETIC_RATE_TABLE} (
                id BIGSERIAL PRIMARY KEY,
                nanostar_id BIGINT NOT NULL UNIQUE
                    REFERENCES {SEQUENCE_TABLE}(id) ON DELETE CASCADE,
                k1 DOUBLE PRECISION,
                k2 DOUBLE PRECISION,
                k3 DOUBLE PRECISION,
                k1m DOUBLE PRECISION,
                k2m DOUBLE PRECISION,
                k3m DOUBLE PRECISION
            )'''
        )
        cursor.execute(
            f'''
                CREATE OR REPLACE FUNCTION delete_nanostar_for_kinetic_rates()
                RETURNS TRIGGER
                LANGUAGE plpgsql
                AS $$
                BEGIN
                    IF pg_trigger_depth() = 1 THEN
                        DELETE FROM {SEQUENCE_TABLE}
                        WHERE id = OLD.nanostar_id;
                    END IF;
                    RETURN OLD;
                END;
                $$;
            '''
        )
        cursor.execute(
            f'''
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM pg_trigger
                        WHERE tgname = 'nanostar_kinetic_rate_delete_sequence'
                    ) THEN
                        CREATE TRIGGER nanostar_kinetic_rate_delete_sequence
                        AFTER DELETE ON {KINETIC_RATE_TABLE}
                        FOR EACH ROW
                        EXECUTE FUNCTION delete_nanostar_for_kinetic_rates();
                    END IF;
                END;
                $$;
            '''
        )
        cursor.execute(
            f'''
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM pg_trigger
                        WHERE tgname = 'nanostar_full_strand_delete_sequence'
                    ) THEN
                        CREATE TRIGGER nanostar_full_strand_delete_sequence
                        AFTER DELETE ON {FULL_STRAND_TABLE}
                        FOR EACH ROW
                        EXECUTE FUNCTION delete_nanostar_for_full_strands();
                    END IF;
                END;
                $$;
            '''
        )


@transaction.atomic
def wipe_nanostar_tables():
    """Delete every nanostar data row and restart IDs."""
    create_nanostar_tables()
    with connection.cursor() as cursor:
        cursor.execute(
            f"TRUNCATE TABLE {KINETIC_RATE_TABLE}, {FULL_STRAND_TABLE}, "
            f"{SEQUENCE_TABLE}, {CURVE_TABLE} "
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
            missing.extend(field for field in FULL_STRAND_FIELDS if field not in row)
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
                    (arm1, arm2, arm3, arm4, middle, linker, "A_Domain",
                     "H", "S", kmeff, keff, curve_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id''',
                [
                    row["arm1"], row["arm2"], row["arm3"], row["arm4"],
                    row["middle"], row["linker"], row["A_Domain"],
                    float(H), float(S), row.get("kmeff"), row.get("keff"),
                    int(curve_id),
                ],
            )
            nanostar_id = cursor.fetchone()[0]
            inserted_ids.append(nanostar_id)
            cursor.execute(
                f'''INSERT INTO {FULL_STRAND_TABLE}
                    (nanostar_id, full_arm1, full_arm2, full_arm3, full_arm4,
                     upper_linker, lower_linker)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)''',
                [
                    nanostar_id,
                    row["full_arm1"], row["full_arm2"],
                    row["full_arm3"], row["full_arm4"],
                    row["upper_linker"], row["lower_linker"],
                ],
            )
            cursor.execute(
                f'''INSERT INTO {KINETIC_RATE_TABLE}
                    (nanostar_id, k1, k2, k3, k1m, k2m, k3m)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)''',
                [
                    nanostar_id,
                    *(row.get(field) for field in KINETIC_RATE_FIELDS),
                ],
            )

    return inserted_ids
