import base64
import io

from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from PIL import Image, UnidentifiedImageError

from .models import Prediction
from .forms import MRIUploadForm


# =========================================================
# HOME PAGE
# =========================================================

def home(request):
    return render(
        request,
        "predictor/home.html"
    )


# =========================================================
# MRI PREDICTION PAGE
# =========================================================

def predict(request):

    if request.method == "POST":

        form = MRIUploadForm(
            request.POST,
            request.FILES
        )

        if form.is_valid():

            uploaded_file = form.cleaned_data["image"]

            # =================================================
            # SAFE IMAGE DECODING
            # =================================================

            try:

                uploaded_file.seek(0)

                image = Image.open(
                    uploaded_file
                )

                image.load()

                image = image.convert(
                    "RGB"
                )

            except (
                UnidentifiedImageError,
                OSError,
                ValueError
            ):

                form.add_error(
                    "image",
                    "Unable to read this image. "
                    "The file may be corrupted or invalid."
                )

                return render(
                    request,
                    "predictor/predict.html",
                    {
                        "form": form
                    }
                )

            # =================================================
            # ONNX PREDICTION
            # =================================================

            try:

                from .ml.inference import predict_image

                result = predict_image(
                    image
                )

                predicted_class = result[
                    "predicted_class"
                ]

                confidence = result[
                    "confidence_percent"
                ]

                probabilities = result[
                    "probabilities"
                ]

            except Exception as error:

                print(
                    "Prediction error:",
                    error,
                    flush=True
                )

                form.add_error(
                    None,
                    "The MRI image could not be analyzed. "
                    "Please try another valid image."
                )

                return render(
                    request,
                    "predictor/predict.html",
                    {
                        "form": form
                    }
                )

            # =================================================
            # ORIGINAL MRI -> BASE64
            # =================================================

            try:

                original_buffer = io.BytesIO()

                resized_image = image.resize(
                    (224, 224)
                )

                resized_image.save(
                    original_buffer,
                    format="PNG",
                    optimize=True
                )

                original_base64 = (
                    base64.b64encode(
                        original_buffer.getvalue()
                    ).decode("utf-8")
                )

                original_buffer.close()

            except Exception as error:

                print(
                    "Image conversion error:",
                    error,
                    flush=True
                )

                form.add_error(
                    None,
                    "Unable to prepare the MRI image "
                    "for display."
                )

                return render(
                    request,
                    "predictor/predict.html",
                    {
                        "form": form
                    }
                )

            # =================================================
            # SCORE-CAM XAI
            # =================================================

            scorecam_base64 = None
            xai_generated = False

            try:

                from .ml.xai import (
                    generate_scorecam,
                    create_scorecam_overlay,
                    image_to_base64,
                )

                class_names = [
                    "Glioma",
                    "Meningioma",
                    "Pituitary"
                ]

                class_idx = class_names.index(
                    predicted_class
                )

                cam, generated_class_idx = (
                    generate_scorecam(
                        image,
                        class_idx=class_idx
                    )
                )

                scorecam_overlay = (
                    create_scorecam_overlay(
                        image,
                        cam
                    )
                )

                scorecam_base64 = (
                    image_to_base64(
                        scorecam_overlay
                    )
                )

                xai_generated = True

                print(
                    "Score-CAM generated successfully.",
                    flush=True
                )

            except Exception as error:

                # Prediction should still work even if
                # explanation generation fails.

                print(
                    "Score-CAM error:",
                    error,
                    flush=True
                )

                scorecam_base64 = None
                xai_generated = False

            # =================================================
            # CLASS PROBABILITIES
            # =================================================

            probability_percent = {

                name: round(
                    value * 100,
                    2
                )

                for name, value
                in probabilities.items()
            }

            # =================================================
            # SAVE RESULT
            # =================================================

            prediction_record = Prediction.objects.create(

                predicted_class=predicted_class,

                confidence=round(
                    confidence,
                    2
                ),

                glioma_probability=probability_percent.get(
                    "Glioma",
                    0
                ),

                meningioma_probability=probability_percent.get(
                    "Meningioma",
                    0
                ),

                pituitary_probability=probability_percent.get(
                    "Pituitary",
                    0
                ),

                xai_generated=xai_generated
            )

            # =================================================
            # RESULT CONTEXT
            # =================================================

            context = {

                "prediction_id":
                    prediction_record.id,

                "predicted_class":
                    predicted_class,

                "confidence":
                    round(
                        confidence,
                        2
                    ),

                "probabilities":
                    probability_percent,

                "original_image":
                    original_base64,

                # New correct name
                "scorecam_image":
                    scorecam_base64,

                "xai_generated":
                    xai_generated,

                # Temporary backward compatibility with
                # existing result.html.
                "gradcam_image":
                    scorecam_base64,
            }

            return render(
                request,
                "predictor/result.html",
                context
            )

    else:

        form = MRIUploadForm()

    return render(
        request,
        "predictor/predict.html",
        {
            "form": form
        }
    )


# =========================================================
# METHODOLOGY PAGE
# =========================================================

def methodology(request):

    return render(
        request,
        "predictor/methodology.html"
    )


# =========================================================
# ABOUT PAGE
# =========================================================

def about(request):

    return render(
        request,
        "predictor/about.html"
    )


# =========================================================
# HISTORY PAGE
# =========================================================

@login_required
def history(request):

    predictions = (
        Prediction.objects
        .all()
        .order_by("-created_at")
    )

    return render(
        request,
        "predictor/history.html",
        {
            "predictions": predictions
        }
    )

