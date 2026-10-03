from django.conf import settings
from django.db import models


class Prediction(models.Model):

    # -----------------------------------------------------
    # USER
    # -----------------------------------------------------

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="predictions",
        null=True,
        blank=True,
    )

    # -----------------------------------------------------
    # UPLOADED MRI IMAGE
    # -----------------------------------------------------

    image = models.ImageField(
        upload_to="prediction_images/",
        null=True,
        blank=True,
    )
    scorecam_image = models.ImageField(
        upload_to="scorecam_images/",
        null=True,
        blank=True,
    )
    # -----------------------------------------------------
    # PREDICTION RESULT
    # -----------------------------------------------------

    predicted_class = models.CharField(
        max_length=50
    )

    confidence = models.FloatField()

    glioma_probability = models.FloatField(
        default=0
    )

    meningioma_probability = models.FloatField(
        default=0
    )

    pituitary_probability = models.FloatField(
        default=0
    )

    xai_generated = models.BooleanField(
        default=False
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    def __str__(self):

        if self.user:
            return (
                f"{self.user.email} - "
                f"{self.predicted_class} - "
                f"{self.confidence:.2f}%"
            )

        return (
            f"{self.predicted_class} - "
            f"{self.confidence:.2f}%"
        )

