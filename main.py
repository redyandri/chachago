from fastapi import FastAPI, File, UploadFile, Form
from starlette.requests import Request
import pickle
import numpy as np
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import random
import uuid
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from fastapi import APIRouter
from fastapi.routing import APIRoute
from typing import Callable
from pydantic import BaseModel
from fastapi import Body
import uvicorn
import datetime
import io
import json
from typing import List
from typing import Dict
from pydantic import BaseModel
import queue
# from docxtpl import DocxTemplate
from io import StringIO
from starlette.responses import StreamingResponse
# import redis
from fastapi.middleware.cors import CORSMiddleware
# from starlette.middleware.cors import CORSMiddleware
# from starlette.middleware import Middleware
import logging
from datetime import datetime
import math
# from neo4j import GraphDatabase
from nltk.tag import CRFTagger
# from decouple import config
import colorsys
import shutil
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ='data/json/proven-cider-382907-953301e26635.json'
from google.cloud import vision
import sys

app = FastAPI()
origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class Response_Item(BaseModel):
    code: str
    activity: str
    level: str
    credit: float


class Responses(BaseModel):
    val: Dict[str, Response_Item]

headers={'Access-Control-Allow-Origin': '*',
'Access-Control-Allow-Methods': '*',
'Access-Control-Allow-Headers': '*'
}

TMP_FILE_REPO='data/tmp/'

vision_client = vision.ImageAnnotatorClient()

@app.post("/")
async def search(request: Request):
    return {'chachago':'greenie bless'}
    
@app.post("/ocr")
async def verify(id: str=Form(...),
                 image: UploadFile = File(...)):
    try:
        # form = await request.form()
        # id=int(form["id"])
        # absent_date=form["absent_date"]     #format '2018-06-29 08:15:27.243860'
        # stream = io.BytesIO(face)
        # data = np.fromstring(stream.getvalue(), dtype=np.uint8)
        # img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        dest_path=os.path.join(TMP_FILE_REPO,image.filename)
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        # return {'filename':image.filename}
        with io.open(dest_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = vision_client.text_detection(image=image)
        text = response.text_annotations[0].description
        text=text.replace('\n',' ')
        os.remove(dest_path)
        return {'text':text}
            
    except Exception as x:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        return {'error':'ERROR======>{},{}'.format(str(x),exc_tb.tb_lineno)}
        
    
