from django.db import models

# Create your models here.


class Prediction(models.Model):
    predicted_class = models.CharField(max_length=50)

    confidence = models.FloatField()

    glioma_probability = models.FloatField(default=0)
    meningioma_probability = models.FloatField(default=0)
    pituitary_probability = models.FloatField(default=0)

    xai_generated = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.predicted_class} - {self.confidence:.2f}%"
    