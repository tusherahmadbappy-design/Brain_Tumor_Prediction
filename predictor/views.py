import base64
import io


from django.contrib.auth.decorators import login_required


from .models import Prediction

from django.shortcuts import render
from PIL import Image, UnidentifiedImageError

from .forms import MRIUploadForm
from .ml.inference import predict_image

from .ml.gradcam import (
    generate_gradcam,
    create_gradcam_overlay,
    image_to_base64,
)



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

    # -----------------------------------------------------
    # POST REQUEST
    # -----------------------------------------------------
    if request.method == "POST":

        form = MRIUploadForm(
            request.POST,
            request.FILES
        )

        # -------------------------------------------------
        # Validate uploaded file
        # -------------------------------------------------
        if form.is_valid():

            uploaded_file = form.cleaned_data["image"]

            # =============================================
            # SAFE IMAGE DECODING
            # =============================================

            try:
                # Reset file pointer
                uploaded_file.seek(0)

                # Open image
                image = Image.open(uploaded_file)

                # Force Pillow to actually decode the image
                image.load()

                # Convert to RGB
                image = image.convert("RGB")

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

            # =============================================
            # BrainGAN-SRNet PREDICTION
            # =============================================

            try:

                result = predict_image(image)

                predicted_class = result[
                    "predicted_class"
                ]

                confidence = result[
                    "confidence_percent"
                ]

                probabilities = result[
                    "probabilities"
                ]

                input_tensor = result[
                    "input_tensor"
                ]

            except Exception as error:

                print(
                    "Prediction error:",
                    error
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

            # =============================================
            # Grad-CAM XAI
            # =============================================

            try:

                cam, class_idx = generate_gradcam(
                    input_tensor
                )

                gradcam_image = (
                    create_gradcam_overlay(
                        image,
                        cam
                    )
                )

            except Exception as error:

                print(
                    "Grad-CAM error:",
                    error
                )

                form.add_error(
                    None,
                    "The prediction was generated, "
                    "but the Grad-CAM explanation "
                    "could not be created."
                )

                return render(
                    request,
                    "predictor/predict.html",
                    {
                        "form": form
                    }
                )

            # =============================================
            # ORIGINAL MRI → BASE64
            # =============================================

            try:

                original_buffer = io.BytesIO()

                resized_image = image.resize(
                    (224, 224)
                )

                resized_image.save(
                    original_buffer,
                    format="PNG"
                )

                original_base64 = (
                    base64.b64encode(
                        original_buffer.getvalue()
                    ).decode("utf-8")
                )

            except Exception as error:

                print(
                    "Image conversion error:",
                    error
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

            # =============================================
            # GRAD-CAM → BASE64
            # =============================================

            try:

                gradcam_base64 = image_to_base64(
                    gradcam_image
                )

            except Exception as error:

                print(
                    "Grad-CAM conversion error:",
                    error
                )

                form.add_error(
                    None,
                    "Unable to prepare the Grad-CAM "
                    "visualization for display."
                )

                return render(
                    request,
                    "predictor/predict.html",
                    {
                        "form": form
                    }
                )

            # =============================================
            # CLASS PROBABILITIES
            # =============================================

            probability_percent = {
                name: round(
                    value * 100,
                    2
                )
                for name, value
                in probabilities.items()
            }

            # =============================================
            # SAVE PREDICTION TO DATABASE
            # =============================================

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

                xai_generated=True
            )

            # =============================================
            # RESULT PAGE CONTEXT
            # =============================================

            context = {
                "prediction_id": prediction_record.id,
                "predicted_class":
                    predicted_class,

                "confidence":
                    round(confidence, 2),

                "probabilities":
                    probability_percent,

                "original_image":
                    original_base64,

                "gradcam_image":
                    gradcam_base64,
            }

            # =============================================
            # SHOW RESULT PAGE
            # =============================================

            return render(
                request,
                "predictor/result.html",
                context
            )

    # -----------------------------------------------------
    # GET REQUEST
    # -----------------------------------------------------
    else:

        form = MRIUploadForm()

    # =====================================================
    # SHOW UPLOAD PAGE
    # =====================================================

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


@login_required
def history(request):

    predictions = Prediction.objects.all().order_by(
        "-created_at"
    )

    return render(
        request,
        "predictor/history.html",
        {
            "predictions": predictions
        }
    )


