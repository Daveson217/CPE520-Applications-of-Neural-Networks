from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.views.generic import TemplateView

# Create your views here.
import cv2
from PIL import Image
from .utils import load_model, predict_emotion

# Load the Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def gen_frames(model_name):
    cap = cv2.VideoCapture(0)
    model = load_model(model_name)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Add the model name at the top-left corner
            cv2.putText(frame, f"Model: {model_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 40, 120), 2)

             # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Crop the face region
                face_region = frame[y:y + h, x:x + w]

                # Convert the frame to PIL Image for processing
                pil_image = Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
                emotion = predict_emotion(model, pil_image)

                # Annotate the frame
                #cv2.putText(frame, f"Model: {model_name} || Emotion: {emotion}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
             # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def live_feed(request, model_name):
    return StreamingHttpResponse(gen_frames(model_name), content_type='multipart/x-mixed-replace; boundary=frame')


class HomePageView(TemplateView):
    template_name = "pages/index.html"
