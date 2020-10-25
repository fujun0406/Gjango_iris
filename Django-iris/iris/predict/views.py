from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from .models import PredResults
# Create your views here.
def predict(request):
    return render(request, 'predict.html')

def predict_chances(request):

    if request.POST.get('action') == 'post':
        method = str(request.POST.get('method'))
        sepal_length = float(request.POST.get('sepal_length'))
        sepal_width = float(request.POST.get('sepal_width'))
        petal_length = float(request.POST.get('petal_length'))
        petal_width = float(request.POST.get('petal_width'))

        model = pd.read_pickle(r"C:\Users\user\Desktop\Linkedin Learning\Django\Django-Iris\{}.pickle".format(method))

        result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

        classification = result[0]
        PredResults.objects.create(sepal_length=sepal_length, sepal_width=sepal_width,
        petal_length=petal_length, petal_width=petal_width, classification=classification)
        #print(request.POST)
        return JsonResponse({'method': method, 'result':classification, 'sepal_length':sepal_length,
        'sepal_width':sepal_width, 'petal_length':petal_length, 'petal_width':petal_width},
        safe = False)

def view_results(request):
    # Submit prediction and show all
    data = {"dataset": PredResults.objects.all()}
    return render(request, "results.html", data)
