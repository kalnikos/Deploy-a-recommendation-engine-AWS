import pandas as pd
from bs4 import BeautifulSoup as bs
import requests
import csv
import urllib.request
import flask
import pickle
import time
from flask import Flask,render_template,url_for,request, redirect
from werkzeug.utils import secure_filename
import os
import docx2txt
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

## Determine a path to save the uploaded CV
UPLOAD_FOLDER = "/home/ubuntu"

app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('home.html')
      
## get the html of the page
## function to extract data
    
@app.route('/results', methods =["GET", "POST"]) 
def results(): 
    def get_features(card):

        try:
            atag = card.h2.a
            job_title = atag.get("title")
        except:
            job_title = None
        try:
            job_url = "http://www.indeed.com" + atag.get("href")
        except:
            job_url = None 
        try:
            new_urls = [x.find('h2', attrs={'class': 'title'}).find('a', href=True)['href'] for x in cards]
            new_urls = "http://www.indeed.com" + atag.get("href")
            page = urllib.request.urlopen(new_urls)
            soup = bs(page.read(), "html.parser")
            job_body = soup.find('div', attrs={'id': "jobDescriptionText"}).text.replace("\n", "")
        except:
            job_body = None

        ## Saving the features to a list    
        feature = (job_title, job_url, job_body)
        #feature = (job_title, job_url, company_name, location, summary, post_date, job_body)
        return feature 

    def preprocess_sentences(text): 
        import nltk 
        nltk.download('punkt') 
        nltk.download('averaged_perceptron_tagger') 
        nltk.download('wordnet') 
        from nltk.stem import WordNetLemmatizer 
        lemmatizer = WordNetLemmatizer() 
        from nltk.corpus import stopwords 
        nltk.download('stopwords') 
        stop_words = set(stopwords.words('english')) 
        VERB_CODES = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
        text = text.lower() 
        temp_sent =[] 
        words = nltk.word_tokenize(text) 
        tags = nltk.pos_tag(words) 
        for i, word in enumerate(words): 
            if tags[i][1] in VERB_CODES:   
                lemmatized = lemmatizer.lemmatize(word, 'v') 
            else: 
                lemmatized = lemmatizer.lemmatize(word) 
            if lemmatized not in stop_words and lemmatized.isalpha(): 
                temp_sent.append(lemmatized) 
        finalsent = ' '.join(temp_sent) 
        finalsent = finalsent.replace("n't", " not") 
        finalsent = finalsent.replace("'m", " am") 
        finalsent = finalsent.replace("'s", " is") 
        finalsent = finalsent.replace("'re", " are") 
        finalsent = finalsent.replace("'ll", " will") 
        finalsent = finalsent.replace("'ve", " have") 
        finalsent = finalsent.replace("'d", " would") 
        return finalsent 
    def text_pre(x):
                ## Applying some text preprocessing
        text = x.replace("\n", "")
        return text
    def extract_email(x):
                ## Extracting the email address using regex  
        import re
        match = re.search(r'[\w\.-]+@[\w\.-]+', x)
        email = match.group(0)
        return email 

    if request.method == "POST": 
       # getting input with name = fname in HTML form 
       job_name = request.form.get("job") 
       # getting input with name = lname in HTML form  
       area_name = request.form.get("area")  
       ## Creating the url of the searching preferation
       file = request.files['file']
       filename = secure_filename(file.filename)
       file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) 
       
       path = Path(filename).absolute() 
       templ = "https://www.indeed.co.uk/jobs?q={}&l={}"    
       url = templ.format(job_name, area_name)
    
       response = requests.get(url)
       soup = bs(response.text, "html.parser")
       features = []
       i = 0
       while i<4:
            try:
                 url = 'https://www.indeed.co.uk' + soup.find("a", {"aria-label": "Next"}).get("href")
            except AttributeError:
                   break
            response = requests.get(url)
            soup = bs(response.text, "html.parser")
            cards = soup.find_all("div", "jobsearch-SerpJobCard")
            for card in cards:            
                feature = get_features(card)
                time.sleep(1)
                features.append(feature)
            i += 1
       data_set = pd.DataFrame(features, columns=["job_title","url", "desc"])
       ## drop duplicates 
       data_set = data_set.drop_duplicates(subset=['url'])
       ## keep only 2 columns
       df = data_set[["url", "desc"]]
       ## Read the candidate's CV and exctract the candidates email and the content
       cv = docx2txt.process(path)
       desc = text_pre(cv)
       url = extract_email(cv)
       ## insert the candidate's CV into the data set
       new_row = {'url':url, 'desc': desc}
       df = df.append(new_row, ignore_index=True)
       # Apply the preprocessing function to the data set
       df["desc"].apply(preprocess_sentences)
       final_data = df[["url", "desc"]]
       ## Find the number of the vacancies that the model has parsed 
       number_of_vacancies = len(data_set)
       
       
       count = CountVectorizer()
       count_matrix = count.fit_transform(final_data['desc'])
       cosine_sim = cosine_similarity(count_matrix)

       def get_index_from_url(url):
           return final_data[final_data["url"] == url].index.values[0]
    
       ul = get_index_from_url(url)
       similar_jobs = list(enumerate(cosine_sim[ul]))
       ## Sort the list
       sorted_similar_jobs = sorted(similar_jobs, key=lambda x:x[1], reverse=True)
       def get_title_from_url(index):
           return final_data[final_data.index == index]["url`"]
           i=0
           for job in sorted_similar_jobs:
              print(get_title_from_url(job[0]))
              i=i+1
              if i>5:
                 break
       df = pd.DataFrame(sorted_similar_jobs[1:10], columns=["index", "similarity"])
       text = []
       for i in df["index"]:
            text.append(final_data["url"].iloc[i])
    
       
       return render_template("results.html", num = str(number_of_vacancies), output = text)
    
if __name__ == '__main__':
    #app.run(debug=True)
    #on ubuntu server
    app.run(host='0.0.0', port=8080)
