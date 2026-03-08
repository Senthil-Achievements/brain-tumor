# tumor_detection/admin.py
from django.contrib import admin
from .models import TumorPrediction

admin.site.register(TumorPrediction)
