"""
URL configuration for BrainGAN_Web project.
"""


from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from predictor.views import CustomPasswordChangeView

urlpatterns = [
    path("admin/", admin.site.urls),

    path(
        "accounts/password/change/",
        CustomPasswordChangeView.as_view(),
        name="account_change_password"
    ),

    path("accounts/", include("allauth.urls")),

    path("", include("predictor.urls")),
]
if settings.DEBUG:
    urlpatterns += static(
        settings.MEDIA_URL,
        document_root=settings.MEDIA_ROOT
    )

