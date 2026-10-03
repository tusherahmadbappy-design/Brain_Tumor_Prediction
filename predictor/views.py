import base64
import io
import uuid

from django.contrib.auth.decorators import login_required
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect, get_object_or_404

from PIL import Image, UnidentifiedImageError

from .models import Prediction
from .forms import MRIUploadForm, ProfileEditForm

from allauth.account.views import PasswordChangeView
from django.urls import reverse_lazy

class CustomPasswordChangeView(PasswordChangeView):
    success_url = reverse_lazy("profile")
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

@login_required
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
            scorecam_saved_file = None
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

                # ---------------------------------------------
                # SCORE-CAM -> BASE64
                # Used for immediate result page.
                # ---------------------------------------------

                scorecam_base64 = (
                    image_to_base64(
                        scorecam_overlay
                    )
                )

                # ---------------------------------------------
                # SCORE-CAM -> SAVED IMAGE
                # Used for prediction history/detail page.
                # ---------------------------------------------

                scorecam_buffer = io.BytesIO()

                scorecam_overlay.save(
                    scorecam_buffer,
                    format="PNG",
                    optimize=True
                )

                scorecam_saved_file = ContentFile(
                    scorecam_buffer.getvalue(),
                    name=(
                        f"scorecam_"
                        f"{uuid.uuid4().hex}.png"
                    )
                )

                scorecam_buffer.close()

                xai_generated = True

                print(
                    "Score-CAM generated successfully.",
                    flush=True
                )

            except Exception as error:

                print(
                    "Score-CAM error:",
                    error,
                    flush=True
                )

                scorecam_base64 = None
                scorecam_saved_file = None
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
            # PREPARE ORIGINAL MRI FOR STORAGE
            # =================================================

            try:

                saved_image_buffer = io.BytesIO()

                image.save(
                    saved_image_buffer,
                    format="PNG",
                    optimize=True
                )

                saved_image_file = ContentFile(
                    saved_image_buffer.getvalue(),
                    name=(
                        f"mri_"
                        f"{uuid.uuid4().hex}.png"
                    )
                )

                saved_image_buffer.close()

            except Exception as error:

                print(
                    "MRI storage preparation error:",
                    error,
                    flush=True
                )

                saved_image_file = None

            # =================================================
            # SAVE RESULT
            # =================================================

            try:

                prediction_record = (
                    Prediction.objects.create(

                        user=request.user,

                        # Original MRI
                        image=saved_image_file,

                        # Historical Score-CAM
                        scorecam_image=scorecam_saved_file,

                        predicted_class=predicted_class,

                        confidence=round(
                            confidence,
                            2
                        ),

                        glioma_probability=(
                            probability_percent.get(
                                "Glioma",
                                0
                            )
                        ),

                        meningioma_probability=(
                            probability_percent.get(
                                "Meningioma",
                                0
                            )
                        ),

                        pituitary_probability=(
                            probability_percent.get(
                                "Pituitary",
                                0
                            )
                        ),

                        xai_generated=xai_generated
                    )
                )

            except Exception as error:

                print(
                    "Database save error:",
                    error,
                    flush=True
                )

                form.add_error(
                    None,
                    "The prediction was completed, but "
                    "the result could not be saved. "
                    "Please try again."
                )

                return render(
                    request,
                    "predictor/predict.html",
                    {
                        "form": form
                    }
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

                "scorecam_image":
                    scorecam_base64,

                "xai_generated":
                    xai_generated,

                # Compatibility with older result template
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
        .filter(
            user=request.user
        )
        .order_by(
            "-created_at"
        )
    )

    return render(
        request,
        "predictor/history.html",
        {
            "predictions": predictions
        }
    )


# =========================================================
# PREDICTION DETAIL
# =========================================================

@login_required
def prediction_detail(
    request,
    prediction_id
):

    prediction = get_object_or_404(
        Prediction,
        id=prediction_id,
        user=request.user
    )

    probabilities = {
        "Glioma":
            prediction.glioma_probability,

        "Meningioma":
            prediction.meningioma_probability,

        "Pituitary":
            prediction.pituitary_probability,
    }

    context = {

        "prediction":
            prediction,

        "predicted_class":
            prediction.predicted_class,

        "confidence":
            prediction.confidence,

        "probabilities":
            probabilities,

        "xai_generated":
            prediction.xai_generated,
    }

    return render(
        request,
        "predictor/prediction_detail.html",
        context
    )


# =========================================================
# USER PROFILE
# =========================================================

@login_required
def profile(request):

    predictions = (
        Prediction.objects
        .filter(
            user=request.user
        )
        .order_by(
            "-created_at"
        )
    )

    return render(
        request,
        "predictor/profile.html",
        {
            "predictions": predictions
        }
    )


# =========================================================
# EDIT PROFILE
# =========================================================

@login_required
def edit_profile(request):

    if request.method == "POST":

        form = ProfileEditForm(
            request.POST,
            instance=request.user
        )

        if form.is_valid():

            form.save()

            return redirect(
                "profile"
            )

    else:

        form = ProfileEditForm(
            instance=request.user
        )

    return render(
        request,
        "predictor/edit_profile.html",
        {
            "form": form
        }
    )


# =========================================================
# DELETE PREDICTION
# =========================================================

@login_required
def delete_prediction(
    request,
    prediction_id
):

    prediction = get_object_or_404(
        Prediction,
        id=prediction_id,
        user=request.user
    )

    if request.method == "POST":

        # ---------------------------------------------
        # DELETE ORIGINAL MRI
        # ---------------------------------------------

        if prediction.image:

            try:

                prediction.image.delete(
                    save=False
                )

            except Exception as error:

                print(
                    "MRI image deletion error:",
                    error,
                    flush=True
                )

        # ---------------------------------------------
        # DELETE SCORE-CAM IMAGE
        # ---------------------------------------------

        if prediction.scorecam_image:

            try:

                prediction.scorecam_image.delete(
                    save=False
                )

            except Exception as error:

                print(
                    "Score-CAM image deletion error:",
                    error,
                    flush=True
                )

        # ---------------------------------------------
        # DELETE DATABASE RECORD
        # ---------------------------------------------

        prediction.delete()

    return redirect(
        "history"
    )

