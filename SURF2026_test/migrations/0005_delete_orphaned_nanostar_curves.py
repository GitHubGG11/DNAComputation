from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("SURF2026_test", "0004_nanostar_tables"),
    ]

    operations = [
        migrations.RunSQL(
            sql='''
                CREATE FUNCTION delete_orphaned_nanostar_curve()
                RETURNS TRIGGER
                LANGUAGE plpgsql
                AS $$
                BEGIN
                    DELETE FROM nanostar_temperature_curves
                    WHERE id = OLD.curve_id
                      AND NOT EXISTS (
                          SELECT 1
                          FROM nanostar_sequences
                          WHERE curve_id = OLD.curve_id
                      );
                    RETURN OLD;
                END;
                $$;

                CREATE TRIGGER nanostar_sequence_delete_curve
                AFTER DELETE ON nanostar_sequences
                FOR EACH ROW
                EXECUTE FUNCTION delete_orphaned_nanostar_curve();
            ''',
            reverse_sql='''
                DROP TRIGGER nanostar_sequence_delete_curve
                    ON nanostar_sequences;
                DROP FUNCTION delete_orphaned_nanostar_curve();
            ''',
        ),
    ]
