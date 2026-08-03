from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("SURF2026_test", "0005_delete_orphaned_nanostar_curves"),
    ]

    operations = [
        migrations.RunSQL(
            sql='''
                CREATE TABLE nanostar_full_strands (
                    id BIGSERIAL PRIMARY KEY,
                    nanostar_id BIGINT NOT NULL UNIQUE
                        REFERENCES nanostar_sequences(id)
                        ON DELETE CASCADE,
                    full_arm1 TEXT NOT NULL,
                    full_arm2 TEXT NOT NULL,
                    full_arm3 TEXT NOT NULL,
                    full_arm4 TEXT NOT NULL,
                    upper_linker TEXT NOT NULL,
                    lower_linker TEXT NOT NULL
                );

                CREATE FUNCTION delete_nanostar_for_full_strands()
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

                CREATE TRIGGER nanostar_full_strand_delete_sequence
                AFTER DELETE ON nanostar_full_strands
                FOR EACH ROW
                EXECUTE FUNCTION delete_nanostar_for_full_strands();
            ''',
            reverse_sql='''
                DROP TRIGGER nanostar_full_strand_delete_sequence
                    ON nanostar_full_strands;
                DROP FUNCTION delete_nanostar_for_full_strands();
                DROP TABLE nanostar_full_strands;
            ''',
        ),
    ]
