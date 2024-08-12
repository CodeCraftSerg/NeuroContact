from django.urls import path, include

from . import views

app_name = "app_predict"

urlpatterns = [
    path("", views.main, name="image_predict"),
]
