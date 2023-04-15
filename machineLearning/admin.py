from django.contrib import admin

# Register your models here.
from .models import ImageModel,Nutrition

# Register your models here.

class ImageAdmin(admin.ModelAdmin):
    list_display = [field.name for field in ImageModel._meta.fields if field.name != "id"] #For displaying every field of model in admin page except id
class NutritionAdmin(admin.ModelAdmin):
    list_display = [field.name for field in Nutrition._meta.fields if field.name != "id"]#For displaying every field of model in admin page except id
    search_fields = ['food_name'] #We can decide which field we can search about
    ordering = ['food_name']

admin.site.register(ImageModel, ImageAdmin)
admin.site.register(Nutrition, NutritionAdmin)