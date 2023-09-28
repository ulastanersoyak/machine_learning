from django.db import models

# Create your models here.

class Nutrition (models.Model):
    food_name = models.CharField(max_length=30)
    protein = models.FloatField()
    carbonhydrate = models.FloatField()
    fat = models.FloatField()

    def __str__(self): #For displaying in admin page
        return self.food_name
    
class ImageModel(models.Model):
    image = models.ImageField(upload_to="images")
    label = models.ForeignKey(Nutrition,on_delete=models.CASCADE , null=True) #If nutrition deleted, label will be deleted too. 

    
