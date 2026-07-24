from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("SURF2026_test", "0003_rename_fit_columns_to_h_s"),
    ]

    operations = [
        migrations.RunSQL(
            sql='''
                CREATE TABLE nanostar_temperature_curves (
                    id BIGSERIAL PRIMARY KEY,
                    curve JSONB NOT NULL
                );

                CREATE TABLE nanostar_sequences (
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
                    curve_id BIGINT NOT NULL
                        REFERENCES nanostar_temperature_curves(id)
                        ON DELETE CASCADE
                );
            ''',
            reverse_sql='''
                DROP TABLE nanostar_sequences;
                DROP TABLE nanostar_temperature_curves;
            ''',
        ),
    ]
