#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 18:53:27 2020

@author: ashwani
"""

import warnings
warnings.filterwarnings("ignore")
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import cv2
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from sklearn.preprocessing import Normalizer, LabelEncoder
import pickle


class FaceDetector:

    def __init__(self):
        self.facenet_model = load_model("facenet_keras.h5")
        self.svm_model = pickle.load(open("SVM_classifier.sav", 'rb'))
        self.data = np.load('faces_dataset_embeddings.npz')
        
        
        self.detector = MTCNN()

    def face_mtcnn_extractor(self, frame):
        
        
        result = self.detector.detect_faces(frame)
        return result

    def face_localizer(self, person):
        
        
        bounding_box = person['box']
        x1, y1 = abs(bounding_box[0]), abs(bounding_box[1])
        width, height = bounding_box[2], bounding_box[3]
        x2, y2 = x1 + width, y1 + height
        return x1, y1, x2, y2, width, height

    def face_preprocessor(self, frame, x1, y1, x2, y2, required_size=(160, 160)):
        
        
        face = frame[y1:y2, x1:x2]
        
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)
        
        face_pixels = face_array.astype('float32')
        
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        
        samples = np.expand_dims(face_pixels, axis=0)
        
        yhat = self.facenet_model.predict(samples)
        face_embedded = yhat[0]
        
        in_encoder = Normalizer(norm='l2')
        X = in_encoder.transform(face_embedded.reshape(1, -1))
        return X

    def face_svm_classifier(self, X):
        
        yhat = self.svm_model.predict(X)
        label = yhat[0]
        yhat_prob = self.svm_model.predict_proba(X)
        probability = round(yhat_prob[0][label], 2)
        trainy = self.data['arr_1']
        
        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)
        predicted_class_label = out_encoder.inverse_transform(yhat)
        label = predicted_class_label[0]
        return label, str(probability)

    def face_detector(self):
        
        cap = cv2.VideoCapture(0)
        while True:
            
            __, frame = cap.read()
            
            result = self.face_mtcnn_extractor(frame)
            if result:
                for person in result:
                    
                    x1, y1, x2, y2, width, height = self.face_localizer(person)
                    
                    X = self.face_preprocessor(frame, x1, y1, x2, y2, required_size=(160, 160))
                    
                    label, probability = self.face_svm_classifier(X)
                    print(" Person : {} , Probability : {}".format(label, probability))
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 155, 255), 2)
                    
                    cv2.putText(frame, label+probability, (x1, height),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
                                lineType=cv2.LINE_AA)
            
            cv2.imshow('frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    facedetector = FaceDetector()
    facedetector.face_detector()