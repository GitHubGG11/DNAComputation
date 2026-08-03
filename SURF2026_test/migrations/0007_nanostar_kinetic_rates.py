from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("SURF2026_test", "0006_nanostar_full_strands"),
    ]

    operations = [
        migrations.RunSQL(
            sql='''
                ALTER TABLE nanostar_sequences
                    ADD COLUMN kmeff DOUBLE PRECISION,
                    ADD COLUMN keff DOUBLE PRECISION;

                CREATE TABLE nanostar_kinetic_rates (
                    id BIGSERIAL PRIMARY KEY,
                    nanostar_id BIGINT NOT NULL UNIQUE
                        REFERENCES nanostar_sequences(id)
                        ON DELETE CASCADE,
                    k1 DOUBLE PRECISION,
                    k2 DOUBLE PRECISION,
                    k3 DOUBLE PRECISION,
                    k1m DOUBLE PRECISION,
                    k2m DOUBLE PRECISION,
                    k3m DOUBLE PRECISION
                );

                INSERT INTO nanostar_kinetic_rates (nanostar_id)
                SELECT id FROM nanostar_sequences;

                CREATE FUNCTION delete_nanostar_for_kinetic_rates()
                RETURNS TRIGGER
                LANGUAGE plpgsql
                AS $$
                BEGIN
                    IF pg_trigger_depth() = 1 THEN
                        DELETE FROM nanostar_sequences
                        WHERE id = OLD.nanostar_id;
                    END IF;
                    RETURN OLD;
                END;
                $$;

                CREATE TRIGGER nanostar_kinetic_rate_delete_sequence
                AFTER DELETE ON nanostar_kinetic_rates
                FOR EACH ROW
                EXECUTE FUNCTION delete_nanostar_for_kinetic_rates();
            ''',
            reverse_sql='''
                DROP TRIGGER nanostar_kinetic_rate_delete_sequence
                    ON nanostar_kinetic_rates;
                DROP FUNCTION delete_nanostar_for_kinetic_rates();
                DROP TABLE nanostar_kinetic_rates;
                ALTER TABLE nanostar_sequences
                    DROP COLUMN keff,
                    DROP COLUMN kmeff;
            ''',
        ),
    ]
