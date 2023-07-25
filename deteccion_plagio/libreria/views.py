from django.shortcuts import render, redirect
from django.http import HttpResponse, FileResponse, HttpResponseRedirect, JsonResponse
from .forms import MiFormulario
from django.contrib import messages
from django.apps import apps
from django.urls import reverse
from datetime import timedelta


from django.conf import settings
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re
from nltk.tokenize import TweetTokenizer

#leer PDF
import PyPDF2

#coseno de similitud
import numpy as np

#word
import docx

#WEB
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googleapiclient.discovery import build
from difflib import SequenceMatcher

#detalle
import os

#divición
from nltk.tokenize import sent_tokenize



def dynamic_threshold(length):
    return 1 / (length ** 0.5)


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)


API_KEY = "AIzaSyA4gkoXlKN4wfEAWyTdFx7shN5pdCT9HqE"
SEARCH_ENGINE_ID = "b3efb1284abf04927"

def process_text(text):

  tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
  text_tokens = tokenizer.tokenize(text)

  stopwords_english = stopwords.words('english')
  text_clean = []

  for word in text_tokens:
      if (word not in stopwords_english): 
          text_clean.append(word)

  return ' '.join(text_clean) 


def search_google(query):
    service = build("customsearch", "v1", developerKey=API_KEY)
    res = service.cse().list(q=query, cx=SEARCH_ENGINE_ID).execute()
    return res.get("items", [])


def get_page_content(url):
    response = requests.get(url)
    return response.text


# Función para calcular la similitud de coseno entre dos listas de oraciones
def cosine_similarity_score(sentences1, sentences2):
    vectorizer = TfidfVectorizer()

    tfidf_matrix1 = vectorizer.fit_transform(sentences1)
    tfidf_matrix2 = vectorizer.transform(sentences2)

    similarity_score = cosine_similarity(tfidf_matrix1, tfidf_matrix2)

    return similarity_score

def split_text_into_sentences(document_content):
    return sent_tokenize(document_content)


# Función para detectar plagio
def detect_plagiarism(document_content, nombre, num_results=100):

    results = []

    document_sentences = split_text_into_sentences(document_content)

    for sentence in document_sentences:

        query = " ".join(sentence.split()[:10])  # Utiliza las primeras 10 palabras del documento como consulta

        search_results = search_google(query)[:num_results]

        for result in search_results:
            url = result["link"]
            page_content = get_page_content(url)
            soup = BeautifulSoup(page_content, "html.parser")
            page_text = soup.get_text()
            page_text = process_text(page_text)

            # Preprocesamiento del contenido de la página web para obtener oraciones individuales
            page_sentences = [sentence.strip() for sentence in page_text.split('.')]

            # Calcular la similitud de coseno entre las oraciones
            similarity_scores = cosine_similarity_score([sentence], page_sentences)

            # obtener el maximo
            max_similarity_score = max(max(similarity_scores.tolist())) #evaluar si seria max o media

            if max_similarity_score > 0.6:  # Ajusta este valor según tus necesidades
                results.append((url, max_similarity_score,nombre))
    
    return results 


def handle_uploaded_file(archivo, ruta_guardado):
    with open(ruta_guardado, 'wb+') as destino:
        for chunk in archivo.chunks():
            destino.write(chunk)









def deteccion(request):

    formulario = MiFormulario(request.POST or None, request.FILES or None)
    resultados = []
    nombres = []
    result = []
    texto = ''
    mensaje = ''

    if formulario.is_valid():
        textoP = formulario.cleaned_data['texto']
        opcion_seleccionada = formulario.cleaned_data['seleccion']
        documentos = request.FILES.getlist('documentos')

        if (opcion_seleccionada == "1"): #plagio
            
            if textoP != "":

                texto = textoP
                nombres.append(texto)
                resultados = detect_plagiarism(texto, "texto")
            
            elif len(documentos) > 0:
                
                for documento in documentos:
                    extension = os.path.splitext(documento.name)[1].lower()
                    ruta_guardado = os.path.join(settings.MEDIA_ROOT, 'documentos', documento.name)
                    handle_uploaded_file(documento, ruta_guardado)
                                    

                    if extension == '.pdf':
                        texto_ = extract_text_from_pdf("documentos/"+documento.name)
                      
                        resultados += detect_plagiarism(texto_, documento.name)
                    
                    elif extension == '.docx':
                        texto_ = extract_text_from_docx("documentos/"+documento.name)

                        resultados += detect_plagiarism(texto_, documento.name)

                    
            else:
                mensaje = "Sube un documento o Ingresa un texto"


        elif (opcion_seleccionada == "0"): #similitud

            if len(documentos) >= 2:

                preprocessed_texts = []

                for documento in documentos:
                    extension = os.path.splitext(documento.name)[1].lower()
                    ruta_guardado = os.path.join(settings.MEDIA_ROOT, 'documentos', documento.name)
                    handle_uploaded_file(documento, ruta_guardado)
                    nombre = documento.name
                    nombres.append(nombre)
                    
                    #extraer textos
                    if extension == '.pdf':
                        texto_ = extract_text_from_pdf("documentos/"+documento.name)
                    
                    elif extension == '.docx':
                        texto_ = extract_text_from_docx("documentos/"+documento.name)

                    #Procesar textos
                    preprocessed_text = process_text(texto_)
                    preprocessed_texts.append(preprocessed_text)

                #obtener vector de textos
                tfidf_vectorizer = TfidfVectorizer()
                tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_texts)

                #obtener vector de textos
                text_lengths = [len(text.split()) for text in preprocessed_texts]


                for i in range(len(documentos)):
                    for j in range(i + 1, len(documentos)):
                        similarity_score = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]

                        # Obtener solo el nombre del archivo o URL sin la ruta completa
                        file_name_i = nombres[i]
                        file_name_j = nombres[j]

                        # Calcular el umbral dinámico para cada texto
                        threshold_i = dynamic_threshold(text_lengths[i])
                        threshold_j = dynamic_threshold(text_lengths[j])

                        static_threshold = 0.6
                        threshold_i = max(static_threshold, threshold_i)
                        threshold_j = max(static_threshold, threshold_j)

                        similarity_percentage = similarity_score * 100

                        # Mostrar el resultado de la comparación
                        if similarity_score > 0.6 or similarity_score > 0.6:     
                            result.append(f"El documento {file_name_i} y el documento {file_name_j} posiblemente tienen plagio, Con un porcentaje de {similarity_percentage:.2f}%")
                        else:
                           result.append(f"El documento {file_name_i} y el documento {file_name_j} no tienen similitud. No hay plagio.")
                           
                        
            else:
                mensaje = "Debes subir al menos 2 documentos"

        
        return render(request, 'paginas/index.html', {'formulario': formulario, 'texto':texto, 'resultados':resultados, 'mensaje': mensaje, 'result':result})

    return render(request, 'paginas/index.html', {'formulario': formulario})





    