
# Register your models here.
from django.contrib import admin
from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):

    list_display = (
        "id",
        "predicted_class",
        "confidence",
        "xai_generated",
        "created_at",
    )

    list_filter = (
        "predicted_class",
        "xai_generated",
        "created_at",
    )

    search_fields = (
        "predicted_class",
    )

    ordering = (
        "-created_at",
    )

    readonly_fields = (
        "created_at",
    )

