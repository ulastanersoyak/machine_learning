from django import forms
from .models import ImageModel


class ImageForm(forms.ModelForm):
    image = forms.ImageField(label="Image")
    class Meta:
        model = ImageModel #Form prepared for ImageModel model 
        fields = ('image',)
        exclude = ['label'] #We want only image as input because we will decide to label
        labels = {
            "image": "Image",
        }
