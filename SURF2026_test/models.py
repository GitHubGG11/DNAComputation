from django.db import models, transaction

# Create your models here.
class SequenceLabel(models.Model):
    label = models.CharField(max_length=200, unique=True)

    def __str__(self):
        return self.label


class SequenceResult(models.Model):
    sequence_label = models.ForeignKey(
        SequenceLabel,
        on_delete=models.CASCADE,
        related_name="results",
    )

    seq1 = models.TextField()
    seq2 = models.TextField()
    energy = models.FloatField()

    class Meta:
        indexes = [
            models.Index(fields=["sequence_label"]),
            models.Index(fields=["energy"]),
        ]

    def __str__(self):
        return f"{self.sequence_label.label}: ({self.seq1}, {self.seq2}) = {self.energy}"
    
def replace_sequence_results(label, results):

    with transaction.atomic():
        sequence_label, _ = SequenceLabel.objects.get_or_create(label=label)

        SequenceResult.objects.filter(sequence_label=sequence_label).delete()

        new_rows = [
            SequenceResult(
                sequence_label=sequence_label,
                seq1=seq1,
                seq2=seq2,
                energy=energy,
            )
            for (seq1, seq2), energy in results
        ]

        SequenceResult.objects.bulk_create(new_rows, batch_size=1000)

