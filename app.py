from flask import Flask, jsonify, request, Response, redirect, url_for
import uuid
from scrapper import Scrapper
import datetime
import sys
import json
from preprocessing.text_preprocessing import TextPreprocessing
from preprocessing.feature_enginering import FeaturesExtraction
from model_dict import get_model
from bertInput import BertInput
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from flask_cors import CORS, cross_origin
from model_dict import models_dic
import pandas as pd
import csv
from werkzeug.utils import secure_filename
import os
import utils

app = Flask(__name__)
CORS(app)

scraper = Scrapper()
sessions = dict()

app.config['UPLOAD_FOLDER'] = "/home/nilsb/INTACT/Interface/IRIT_TEXT_Backend_Forked/outputs/uploaded_files"

@app.route('/api/get_noFeatures_models/', methods=['GET'])
@cross_origin()
def get_noFeatures_models():
    dict_ = {}
    for Field,class_task in models_dic.items(): 
        dict_[Field]=[]
        for i, k in enumerate(class_task):
            dict_[Field].append({})
            dict_[Field][i]['name'] = k
            dict_[Field][i]['models'] = []
            j = 0

            for k_, v_ in models_dic[Field][k]['models'].items():
                if 'features' not in models_dic[Field][k]['models'][k_]:
                    dict_[Field][i]['models'].append({'name': k_})
    response = jsonify(dict_)
    return response


@app.route('/api/get_all_models/', methods=['GET'])
@cross_origin()
def get_all_models():

    dict_ = {}

    for Field,class_task in models_dic.items(): 
        dict_[Field]=[]
        for i, k in enumerate(class_task):
            dict_[Field].append({})
            dict_[Field][i]['name'] = k
            dict_[Field][i]['models'] = [{'name': v_} for k_,
                                v_ in enumerate(models_dic[Field][k]['models'])]

    response = jsonify(dict_)
    return response


@app.route('/api/tweet_data/', methods=['POST'])
@cross_origin()
def scrap(ID):
    if request.method == 'POST':
        ID = request.json['ID']
        tweet = scraper.get_tweet_byID(ID)
        return jsonify(tweet)


@app.route('/api/scrap_tweets/', methods=['POST'])
@cross_origin()
def scrap_df():
    if request.method == 'POST':
        # we will get the file from the request
        print(request.json)
        data = request.json
        session_token = data['session_token']
        print(session_token)
        if session_token == "null":
            print("Génération nouveau id")
            session_token = str(uuid.uuid4())
        session_token = str(uuid.uuid4())
        lang = 'fr'
        limit = int(data["limit_scrap"])
        begin_date = datetime.datetime.strptime(data["begin_date"],
                                                "%d/%m/%Y").date()
        end_date = datetime.datetime.strptime(data["end_date"],
                                              "%d/%m/%Y").date()
        keywords = data["keywords"]
        keywords = [str(r) for r in keywords]  # Remove encoding
        df = scraper.get_tweets_df(keywords=keywords,
                                   lang=lang,
                                   begindate=begin_date,
                                   enddate=end_date,
                                   limit=limit)
        addInstances(session_token,sessions,df)
        print(df.head)
        response = jsonify({
            'session_token': session_token,
            'dataframe_length': df.shape[0]
        })
        return response


@app.route("/api/predict_dataframe", methods=["POST"])
@cross_origin()
def predict():
    if request.method == 'POST':
        data = request.json
        print(sessions)
        session_token = str(data["session_token"])
        model_name = str(data["model_name"])
        domain_name = str(data["field"])
        session_token = session_token
        print(session_token)
        df = sessions[session_token]
        # Feature enginnering

        featuresExtrator = FeaturesExtraction(df, "text")
        featuresExtrator.fit_transform()

        # Preprocessing
        text_preprocessing = TextPreprocessing(df, "text")
        text_preprocessing.fit_transform()
        print(df["text"])
        # drop small-text columns
        #TODO Check length minimal
        # df = df[~(df['text'].str.len() > 60)]
        # Load model ,Tokenizer , labels_dict , features

        model, tokenizer, labels_dict, features = get_model(
            domain_name, model_name)

        # get text
        sentences = df["processed_text"]
        bert_input = BertInput(tokenizer)
        sentences = bert_input.fit_transform(sentences)
        input_ID = torch.tensor(sentences[0])
        input_MASK = torch.tensor(sentences[1])
        print(len(sentences))
        if features:
            features_column = df[features].values.astype(float).tolist()
            features_column = torch.tensor(features_column)
            tensor_dataset = TensorDataset(
                input_ID, input_MASK, features_column)
        else:
            tensor_dataset = TensorDataset(input_ID, input_MASK)
        dataloader = DataLoader(
            tensor_dataset, batch_size=1, shuffle=False, num_workers=4)

        pred = []
        for index, batch in enumerate(dataloader):
            output = model(batch)
            label_index = np.argmax(output[0].cpu().detach().numpy())
            print(index)
            pred.append(labels_dict.get(label_index))
        df['prediction'] = pred

        # for label in 
        # df1['TEXTE'][df1['urgence'] == "Message-InfoUrgent"]

    
        # Inference
        response = jsonify({
            'session_token': session_token,
            'dataframe': df.to_json(orient="records"),
            'allLabels' : json.dumps(df['prediction'].unique().tolist()),
            'summary': df['prediction'].value_counts().to_json(),
        })
        return response


@app.route("/api/get_data", methods=["POST"])
@cross_origin()
def get_data():
    print("Function get dataAPI")
    if request.method == 'POST':
        data = request.json
        session_token = str(data["session_token"])
        if session_token in sessions.keys():
            df = sessions[session_token]
            data =df.to_json(orient="records")
            word_cloud_data = utils.get_data_word_cloud(df['text'])
            print(word_cloud_data)
            word_cloud_data = json.dumps(word_cloud_data)
        else:
            data = "no_data"
            word_cloud_data = "no_data"
        
        response = jsonify({
            'session_token': session_token,
            'dataframe': data,
            'word_cloud_data' :word_cloud_data 
        })
        return response

@app.route("/api/predict_onetweet", methods=["POST"])
@cross_origin()
def predict_one():
    if request.method == 'POST':
        data = request.json
        print(data)
        model_name = str(data["model_name"])
        domain_name = str(data["field"])
        df = pd.DataFrame.from_dict({"text": [data['text']]})
        print(data['text'])



        # Feature enginnering
        print(df)
        featuresExtrator = FeaturesExtraction(df, "text")
        featuresExtrator.fit_transform()

        # Preprocessing
        text_preprocessing = TextPreprocessing(df, "text")
        text_preprocessing.fit_transform()

        # drop small-text columns
        print(len(df))
        df = df[~(df['processed_text'].str.len() > 100)]
        print(len(df['processed_text'][0]))
        print(len(df))
        #df = df[len(df['processed_text']) > 60]
        # Load model ,Tokenizer , labels_dict , features

        model, tokenizer, labels_dict, features = get_model(
            domain_name, model_name)

        # get text
        sentences = df["processed_text"]
        bert_input = BertInput(tokenizer)
        sentences = bert_input.fit_transform(sentences)
        input_ID = torch.tensor(sentences[0])
        input_MASK = torch.tensor(sentences[1])
        print(len(sentences))

        pred = []
        output = model((input_ID, input_MASK,))
        label_index = np.argmax(output[0].cpu().detach().numpy())
        pred.append(labels_dict.get(label_index))
        df['prediction'] = pred

        print(df['prediction'].iloc[0])
        # Inference
        response = jsonify({
            'prediction': df['prediction'].iloc[0],
            'allLabels': list(labels_dict.values())
        })

        return response


@app.route("/api/save_correction", methods=["POST"])
@cross_origin()
def save_correction():
    if request.method == 'POST':
        data = request.json
        print(data)
        row = [data['tweetID'],data['text'], data['labelPred'], data['correctedLabel']]
        filename = "outputs/manual_correction/corrections.csv"
        with open(filename, 'a+',newline='') as csvfile: 
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(row)        
        return jsonify("OK")


@app.route("/api/uploadCSV", methods=["POST"])
@cross_origin() 
def uploadCSV():
    if request.method == 'POST':
        session_token = request.form['session_token']
        if session_token == "null":
            session_token = str(uuid.uuid4())    
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and utils.allowed_file(file.filename, ['csv']):
            col_text = request.form['text_col']
            filename = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(file.filename))
            file.save(filename)
            df =pd.read_csv(filename)
            df = df.rename(columns={col_text:'text'})
            
            all_cols_but_text = [col for col in utils.COL_USED if col!='text'] 
            for col in  all_cols_but_text:
                df[col] ='unknown'
            
            print(df.head())
            addInstances(session_token,sessions,df[utils.COL_USED])
    response = jsonify({
        'session_token': session_token,
        'dataframe_length': df.shape[0]
    })
    return response


def addInstances(id,dict,df):
    print("Add instance ")
    if id in dict.keys():
        prec = dict[id] 
        dict[id] = pd.concat([prec,df])
        print("rajout")
        print(dict[id])
    else:
        print("New")
        dict[id] = df
    

if __name__ == "__main__"   :
    app.run(debug=True, host='127.0.0.1', port=4000)