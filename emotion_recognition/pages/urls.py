from django.urls import path
from . import views


urlpatterns = [
    path('', views.HomePageView.as_view()),
    path('live_feed/<str:model_name>/', views.live_feed, name='live_feed'),   
]