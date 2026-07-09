from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("SURF2026_test", "0002_linkertemperaturecurve_validlinkersequence"),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[
                migrations.RunSQL(
                    sql=(
                        'ALTER TABLE valid_linker_sequences '
                        'RENAME COLUMN "slope_kcal_per_mol_K" TO "S"'
                    ),
                    reverse_sql=(
                        'ALTER TABLE valid_linker_sequences '
                        'RENAME COLUMN "S" TO "slope_kcal_per_mol_K"'
                    ),
                ),
                migrations.RunSQL(
                    sql='UPDATE valid_linker_sequences SET "S" = -"S"',
                    reverse_sql='UPDATE valid_linker_sequences SET "S" = -"S"',
                ),
                migrations.RunSQL(
                    sql=(
                        'ALTER TABLE valid_linker_sequences '
                        'RENAME COLUMN "intercept_kcal_per_mol" TO "H"'
                    ),
                    reverse_sql=(
                        'ALTER TABLE valid_linker_sequences '
                        'RENAME COLUMN "H" TO "intercept_kcal_per_mol"'
                    ),
                ),
            ],
            state_operations=[
                migrations.RemoveField(
                    model_name="validlinkersequence",
                    name="slope_kcal_per_mol_K",
                ),
                migrations.RemoveField(
                    model_name="validlinkersequence",
                    name="intercept_kcal_per_mol",
                ),
                migrations.AddField(
                    model_name="validlinkersequence",
                    name="H",
                    field=models.FloatField(db_column="H"),
                ),
                migrations.AddField(
                    model_name="validlinkersequence",
                    name="S",
                    field=models.FloatField(db_column="S"),
                ),
            ],
        ),
    ]
