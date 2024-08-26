from django.urls import path
from app_datascience.views import data_science_page

urlpatterns = [
    path('data-science/', data_science_page, name='data_science_page'),
]
