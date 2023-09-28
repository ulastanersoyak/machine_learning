from django.urls import path 
from . import views


urlpatterns = [
    path("prediction-result", views.prediction_page.as_view(), name="prediction-page"),
    path("prediction-page", views.prediction_page.as_view(), name="prediction-page"),
    path("all-predictions", views.all_predictions_page, name="all-predictions"),
    path("about-page", views.about_page, name="about-page")
]
