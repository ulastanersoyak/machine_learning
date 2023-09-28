from django.shortcuts import render
from django.views import View
from .forms import ImageForm
from .models import ImageModel, Nutrition
from .food_model import *
import torchvision
from torch import nn

model = torchvision.models.efficientnet_b0() #declaring our model
model.classifier[1] = nn.Linear(in_features=1280,out_features=101,bias=True) #Changing last layers out_features of the model because we have 101 classes
model.load_state_dict(torch.load("machineLearning/food101model.pth",map_location=torch.device("cpu"))) #Loading our trained parameters to model
model.eval() 

class prediction_page(View):
 
    def get(self, request): #will be excuted when someone refresh or get to the page
        form = ImageForm()
        return render(request, "machineLearning/prediction-page.html", {
            "form": form
        })

    def post(self, request): # will be executed if user upload image
        
        form = ImageForm(request.POST , request.FILES)

        if form.is_valid(): #Checking the file if it is an image or not.

            image_to_upload = ImageModel(image=request.FILES["image"])
            image_to_upload.save() #Saving model first to obtain image from user

            returned_tuple=prediction(img_path=image_to_upload.image.path,model=model) #returns class name and top 5 probabilities
            pred=returned_tuple[0] #declaring class name
            nutritions =Nutrition.objects.get(food_name=pred) #getting nutrition rates by class name

            imageModel=ImageModel.objects.get(image=image_to_upload.image) #getting image to update its label
            
            imageModel.label=nutritions #updating the image's label

            imageModel.save()#saving updated model
            
            #declaring nutrition rates for viewing in template
            protein=nutritions.protein
            fat=nutritions.fat
            carbonhydrate=nutritions.carbonhydrate

            return render(request,"machineLearning/prediction-result.html",{
                "image":imageModel.image,
                "class":pred,
                "protein":protein,
                "fat":fat,
                "carbonhydrate":carbonhydrate,
                "returned_tuple":returned_tuple[1]}
                )

        return render(request, "machineLearning/prediction-page.html", {
            "form": form
        })

def all_predictions_page(request):

    all_preds = ImageModel.objects.all().order_by('-id')[:4] #Getting last 4 image label pair
    
    return render(request,"machineLearning/all-predictions.html",{"all_preds":all_preds})

def about_page(request):
    return render(request,"machineLearning/about-page.html")