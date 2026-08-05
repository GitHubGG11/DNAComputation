"""Extract nanostar arm sequences from the screening workbook.

The legacy ``.xls`` format requires the third-party ``xlrd`` package:
    python -m pip install xlrd
"""

from pathlib import Path
from typing import Union

import pandas as pd


DEFAULT_WORKBOOK = Path(r"C:\Users\GeneH\Downloads\Plate_nodye_16orthogonal (1).xls")
SEQUENCE_COLUMN = "Oligo sequence (5' to 3')"


def get_ns_arms(workbook: Union[str, Path] = DEFAULT_WORKBOOK) -> list[str]:
    """Return the 20-base NS arm from every nonblank sequence in column F.

    The arm retains the sequence's 5'-to-3' orientation. Its final base is the
    tenth base counted from the sequence's end (equivalent to ``seq[-29:-9]``).
    """
    workbook = Path(workbook)
    excel_file = pd.ExcelFile(workbook, engine="xlrd")
    table = None
    for sheet_name in excel_file.sheet_names:
        candidate = pd.read_excel(excel_file, sheet_name=sheet_name)
        if SEQUENCE_COLUMN in candidate.columns:
            table = candidate
            break

    if table is None:
        raise ValueError(
            f"Could not find a sheet containing {SEQUENCE_COLUMN!r} in {workbook}."
        )

    sequences = table[SEQUENCE_COLUMN]

    arms: list[str] = []
    for spreadsheet_row, value in enumerate(sequences, start=2):
        if pd.isna(value) or not str(value).strip():
            continue

        sequence = "".join(str(value).split())
        if len(sequence) < 29:
            raise ValueError(
                f"Row {spreadsheet_row} has only {len(sequence)} characters; "
                "at least 29 are required."
            )
        arms.append(sequence[-29:-9])

    return arms


if __name__ == "__main__":
    for arm in get_ns_arms():
        print(arm)
