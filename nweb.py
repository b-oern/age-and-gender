from age_and_gender import *
from PIL import Image, ImageDraw, ImageFont
from nwebclient import NWebClient
import os

data = AgeAndGender()
data.load_shape_predictor('example/models/shape_predictor_5_face_landmarks.dat')
data.load_dnn_gender_classifier('example/models/dnn_gender_classifier_v1.dat')
data.load_dnn_age_predictor('example/models/dnn_age_predictor_v1.dat')

def mapDoc(doc, client):
    filename = 'current.jpg'  
    doc.downloadThumbnail(file=filename, size='m')
    img = Image.open(filename).convert("RGB")
    result = data.predict(img)
    print(result)
    return str(result)


n = NWebClient(os.environ['NWEB_URL'], os.environ['NWEB_USER'], os.environ['NWEB_PASS'])
n.mapDocMeta('ml', 'age_and_gender', filterArgs='kind=image', limit=500, mapFunction=mapDoc)
