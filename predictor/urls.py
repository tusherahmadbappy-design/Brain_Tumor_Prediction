from django.urls import path
from . import views


urlpatterns = [

    # =====================================================
    # HOME
    # =====================================================

    path(
        "",
        views.home,
        name="home"
    ),


    # =====================================================
    # MRI ANALYSIS
    # =====================================================

    path(
        "predict/",
        views.predict,
        name="predict"
    ),


    # =====================================================
    # METHODOLOGY
    # =====================================================

    path(
        "methodology/",
        views.methodology,
        name="methodology"
    ),


    # =====================================================
    # ABOUT
    # =====================================================

    path(
        "about/",
        views.about,
        name="about"
    ),


    # =====================================================
    # HISTORY
    # =====================================================

    path(
        "history/",
        views.history,
        name="history"
    ),


    # =====================================================
    # VIEW SAVED PREDICTION
    # =====================================================

    path(
        "history/<int:prediction_id>/",
        views.prediction_detail,
        name="prediction_detail"
    ),


    # =====================================================
    # DELETE PREDICTION
    # =====================================================

    path(
        "history/<int:prediction_id>/delete/",
        views.delete_prediction,
        name="delete_prediction"
    ),


    # =====================================================
    # USER PROFILE
    # =====================================================

    path(
        "profile/",
        views.profile,
        name="profile"
    ),


    # =====================================================
    # EDIT PROFILE
    # =====================================================

    path(
        "profile/edit/",
        views.edit_profile,
        name="edit_profile"
    ),
]
