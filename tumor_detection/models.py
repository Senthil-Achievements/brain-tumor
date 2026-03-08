
from django.db import models

class TumorPrediction(models.Model):
    TUMOR_CHOICES = [
        ('glioma', 'Glioma Tumor'),
        ('meningioma', 'Meningioma Tumor'),
        ('no_tumor', 'No Tumor'),
        ('pituitary', 'Pituitary Tumor'),
    ]

    # Store uploaded MRI image
    image = models.ImageField(upload_to='tumor_images/')
    
    # Store predicted tumor type
    predicted_class = models.CharField(max_length=20, choices=TUMOR_CHOICES)
    
    # Store confidence score (e.g., percentage of prediction confidence)
    confidence_score = models.FloatField()
    
    # Date and time of prediction
    prediction_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.predicted_class} - {self.confidence_score:.2f}% confidence"

