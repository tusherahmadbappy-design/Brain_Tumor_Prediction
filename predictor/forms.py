from django import forms
from PIL import Image, UnidentifiedImageError


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
    