from django.shortcuts import render


from django.shortcuts import render

def list(request):
    return render(request, 'list.html', {})

def home(request):
    return render(request, 'home.html', {})