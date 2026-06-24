from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="simulation_index"),
    path("stream/", views.stream_simulation, name="stream_simulation"),
]