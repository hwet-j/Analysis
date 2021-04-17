from django.contrib import admin
from django.urls import path,include
from django_covid import views

urlpatterns = [
    path('', views.home, name="home"),
    path('list', views.list, name="list"),
    path('list_gender', views.list_gender, name="list_gender"),
    path('list_vaccin', views.list_vaccin, name="list_vaccin")
]
