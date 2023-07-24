from django.urls import path
from . import views

from django.conf import settings
from django.contrib.staticfiles.urls import static

urlpatterns = [
    #pagina principal
    path("", views.deteccion, name="deteccion"),

]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)