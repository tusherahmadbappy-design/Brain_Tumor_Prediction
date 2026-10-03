from django import forms
from django.contrib.auth import get_user_model
from PIL import Image, UnidentifiedImageError


User = get_user_model()


# =========================================================
# MRI UPLOAD FORM
# =========================================================

class MRIUploadForm(forms.Form):

    image = forms.ImageField(
        label="Upload Brain MRI",
        help_text="Supported formats: JPG, JPEG, PNG"
    )

    def clean_image(self):

        image = self.cleaned_data.get("image")

        if not image:
            raise forms.ValidationError(
                "Please select a brain MRI image."
            )

        # Maximum file size = 10 MB
        if image.size > 10 * 1024 * 1024:
            raise forms.ValidationError(
                "The uploaded image is too large. "
                "Maximum allowed size is 10 MB."
            )

        allowed_content_types = [
            "image/jpeg",
            "image/png",
        ]

        if image.content_type not in allowed_content_types:
            raise forms.ValidationError(
                "Unsupported file type. "
                "Please upload a JPG, JPEG, or PNG image."
            )

        # Verify actual image content
        try:

            img = Image.open(image)

            img.verify()

            image.seek(0)

        except (
            UnidentifiedImageError,
            OSError,
            ValueError
        ):

            raise forms.ValidationError(
                "The uploaded file is not a valid image "
                "or the image is corrupted."
            )

        return image


# =========================================================
# PROFILE EDIT FORM
# =========================================================

class ProfileEditForm(forms.ModelForm):

    first_name = forms.CharField(
        max_length=150,
        required=True,
        label="First Name",
        widget=forms.TextInput(
            attrs={
                "placeholder": "Enter your first name",
                "class": "profile-input",
            }
        ),
    )

    last_name = forms.CharField(
        max_length=150,
        required=True,
        label="Last Name",
        widget=forms.TextInput(
            attrs={
                "placeholder": "Enter your last name",
                "class": "profile-input",
            }
        ),
    )

    email = forms.EmailField(
        required=True,
        label="Email Address",
        widget=forms.EmailInput(
            attrs={
                "placeholder": "Enter your email address",
                "class": "profile-input",
            }
        ),
    )

    class Meta:

        model = User

        fields = [
            "first_name",
            "last_name",
            "email",
        ]

    # -----------------------------------------------------
    # Prevent two accounts from using the same email
    # -----------------------------------------------------

    def clean_email(self):

        email = self.cleaned_data[
            "email"
        ].strip().lower()

        email_exists = (
            User.objects
            .filter(
                email__iexact=email
            )
            .exclude(
                pk=self.instance.pk
            )
            .exists()
        )

        if email_exists:

            raise forms.ValidationError(
                "An account with this email address "
                "already exists."
            )

        return email
    