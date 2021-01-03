#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 18:22:18 2020

@author: ashwani
"""
# import libraries
import warnings
warnings.filterwarnings("ignore")
import datetime
import time
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import savez_compressed, asarray, load, expand_dims
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle



class FaceTrainer:

    def __init__(self):
        self.dataset_train = "/home/rapidadmin/Desktop/SVM_classification/train/"
        self.dataset_val = "/home/rapidadmin/Desktop/SVM_classification/val/"
        self.faces_npz = "faces_dataset.npz"
        self.keras_facenet = "facenet_keras.h5"
        self.faces_embeddings = "faces_dataset_embeddings.npz"
        self.svm_classifier = "SVM_classifier.sav"
        return

    def load_dataset(self, directory):
   
        X = []
        y = []
        
        for subdir in listdir(directory):
            path = directory + subdir + '/'
            
            if not isdir(path):
                continue
            
            faces = self.load_faces(path)
            
            labels = [subdir for _ in range(len(faces))]
            print("loaded {} examples for class: {}".format(len(faces), subdir))
            X.extend(faces)
            y.extend(labels)
        return asarray(X), asarray(y)

    def load_faces(self, directory):
        
        faces = []
        
        for filename in listdir(directory):
            path = directory + filename
            
            face = self.extract_face(path)
            faces.append(face)
        return faces

    def extract_face(self, filename, required_size=(160, 160)):
        
        image = Image.open(filename)
       
        image = image.convert('RGB')
       
        pixels = asarray(image)
       
        detector = MTCNN()
        
        results = detector.detect_faces(pixels)
        
        x1, y1, width, height = results[0]['box']
        
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
       
        face = pixels[y1:y2, x1:x2]
       
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array

    def create_faces_npz(self):
        
        trainX, trainy = self.load_dataset(self.dataset_train)
        print("Training data set loaded")
        
        testX, testy = self.load_dataset(self.dataset_val)
        print("Testing data set loaded")
       
        savez_compressed(self.faces_npz, trainX, trainy, testX, testy)
        return

    def create_faces_embedding_npz(self):
        
        data = load(self.faces_npz)
        trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
       
        model = load_model(self.keras_facenet)
        print('Keras Facenet Model Loaded')
       
        newTrainX = list()
        for face_pixels in trainX:
            embedding = self.get_embedding(model, face_pixels)
            newTrainX.append(embedding)
        newTrainX = asarray(newTrainX)
      
        newTestX = list()
        for face_pixels in testX:
            embedding = self.get_embedding(model, face_pixels)
            newTestX.append(embedding)
        newTestX = asarray(newTestX)
       
        savez_compressed(self.faces_embeddings, newTrainX, trainy, newTestX, testy)
        return

    def get_embedding(self, model, face_pixels):
       
        face_pixels = face_pixels.astype('float32')
        
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
      
        samples = expand_dims(face_pixels, axis=0)
     
        yhat = model.predict(samples)
        return yhat[0]

    def classifier(self):
       
        data = load(self.faces_embeddings)
        trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
        
        in_encoder = Normalizer(norm='l2')
        trainX = in_encoder.transform(trainX)
        testX = in_encoder.transform(testX)
       
        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)
        trainy = out_encoder.transform(trainy)
        testy = out_encoder.transform(testy)
       
        model = SVC(kernel='linear', probability=True)
        model.fit(trainX, trainy)
       
        filename = self.svm_classifier
        pickle.dump(model, open(filename, 'wb'))
        
        yhat_train = model.predict(trainX)
        yhat_test = model.predict(testX)
       
        score_train = accuracy_score(trainy, yhat_train)
        score_test = accuracy_score(testy, yhat_test)
      
        print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))
        return

    def start(self):
        
        start_time = time.time()
        st = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        print("-----------------------------------------------------------------------------------------------")
        print("Face trainer Initiated at {}".format(st))
        print("-----------------------------------------------------------------------------------------------")
       
        self.create_faces_npz()
        
        self.create_faces_embedding_npz()
        
        self.classifier()
        end_time = time.time()
        et = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        print("-----------------------------------------------------------------------------------------------")
        print("Face trainer Completed at {}".format(et))
        print("Total time Elapsed {} secs".format(round(end_time - start_time), 0))
        print("-----------------------------------------------------------------------------------------------")

        return


if __name__ == "__main__":
    facetrainer = FaceTrainer()
    facetrainer.start()