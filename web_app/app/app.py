from operator import mod
from flask import Flask, render_template, request
import numpy as np
from joblib import dump, load
from csv import writer
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

app = Flask(__name__)

model = load('model.joblib')


@app.route('/', methods=['GET', 'POST'])
def home_page():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        retrain_model()
        return render_template('index.html')


@app.route("/oznacevanje", methods=['GET', 'POST'])
def oznacevanje():
    if request.method == 'GET':
        return render_template("oznacevanje.html")
    else:
        vreme = request.form['vreme']
        dan = request.form['dan']
        se_mudi = request.form['mudi']
        pocutje = request.form['pocutje']
        emotikon = request.form['emotikon']

        new_row = [vreme, dan, se_mudi, pocutje, emotikon]
        write_to_csv(new_row)

        return render_template("oznacevanje.html")


@app.route("/prikaz", methods=['GET', 'POST'])
def prikaz():
    if request.method == 'GET':
        return render_template("prikaz.html", show_emoji=0)
    else:
        vreme = request.form['vreme']
        se_mudi = request.form['mudi']
        pocutje = request.form['pocutje']
        dan = request.form['dan']

        vreme_value = vreme_to_value(vreme)
        se_mudi_value = mudi_to_value(se_mudi)
        pocutje_value = pocutje_to_value(pocutje)
        dan_value = dan_to_value(dan)

        model_input = np.array(
            [[vreme_value, dan_value, se_mudi_value, pocutje_value]])
        prediction = model.predict(model_input)
        prediction_value = prediction[0]
        emoji_string = pred_value_to_emoji_string(prediction_value)

        return render_template("prikaz.html", emoji=emoji_string, show_emoji=1)


def dan_to_value(pocutje):
    if pocutje == 'ponedeljek':
        return 2
    if pocutje == 'torek':
        return 4
    if pocutje == 'sreda':
        return 3
    if pocutje == 'cetrtek':
        return 0
    else:
        return 1


def pocutje_to_value(pocutje):
    if pocutje == 'dobro':
        return 0
    else:
        return 1


def vreme_to_value(vreme):
    if vreme == "oblacno":
        return 1
    if vreme == "soncno":
        return 2
    else:
        return 0


def mudi_to_value(se_mudi):
    if se_mudi == 'da':
        return 0
    else:
        return 1


def pred_value_to_emoji_string(value):
    if value == -3:
        return "1F622"
    if value == -2:
        return "1F641"
    if value == -1:
        return "1F610"
    if value == 1:
        return "1F642"
    if value == 2:
        return "1F600"
    else:
        return "1F604"


"""
def postaja_to_value(postaja):
    postaja_l = postaja.lower()
    if postaja_l == "konzorcij":
        return 0
    if postaja_l == "gosposvetska":
        return 1
    if postaja_l == "hajdrihova":
        return 2
    if postaja_l == "jamnikarjeva":
        return 3
    if postaja_l == "":
        return 4
    if postaja_l == "":
        return 5
    if postaja_l == "":
        return 6
    if postaja_l == "":
        return 7
    if postaja_l == "":
        return 8
    if postaja_l == "":
        return 9
    if postaja_l == "":
        return 10
    if postaja_l == "":
        return 11
    if postaja_l == "ajdovscina":
        return 12 
"""


def write_to_csv(row):
    with open('data.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(row)
        f_object.close()


def retrain_model():
    le = preprocessing.LabelEncoder()
    df = pd.read_csv('data.csv')

    le.fit(df['vreme'])
    transformed_vreme = le.transform(df['vreme'])
    df['vreme'] = transformed_vreme

    le.fit(df['se_mudi'])
    transformed_mudi = le.transform(df['se_mudi'])
    df['se_mudi'] = transformed_mudi

    le.fit(df['dan'])
    transformed_dan = le.transform(df['dan'])
    df['dan'] = transformed_dan

    le.fit(df['pocutje'])
    transformed_pocutje = le.transform(df['pocutje'])
    df['pocutje'] = transformed_pocutje

    X = df.drop(columns=['eid'])
    Y = df['eid']

    model = DecisionTreeClassifier()
    model.fit(X.values, Y)

    dump(model, 'model.joblib')

    return


"""
@app.route("/")
def hello_world():
    test_np_input = np.array([[69, 1, 1]])
    model = load('model.joblib')
    preds = model.predict(test_np_input)
    preds_as_str = str(preds)
    return preds_as_str
"""
