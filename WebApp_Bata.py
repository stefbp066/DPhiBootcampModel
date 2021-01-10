#!/usr/bin/env python
# coding: utf-8

# In[1]:

from datetime import datetime
start_time = datetime.now()

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,scale,StandardScaler
import xgboost as xgb
import logging


# In[2]:

loan_data = pd.read_csv("loan_train.csv",index_col='Unnamed: 0')
logging.warning("Total waktu sampai proses pembacaan data: "+str(datetime.now()-start_time))
loan_data = loan_data.drop(['Loan_ID'],axis=1)
for col in loan_data.select_dtypes(['float64','int64']):
    loan_data[col] = loan_data[col].fillna(loan_data[col].median())
col_kat = list(loan_data.select_dtypes(include='object').columns)
im_Gender = SimpleImputer(missing_values=np.NaN,strategy='most_frequent')
im_Gender = im_Gender.fit(loan_data[['Gender']])
loan_data.Gender = im_Gender.transform(loan_data[['Gender']]).ravel()

im_Married = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
im_Married = im_Married.fit(loan_data[['Married']])
loan_data['Married'] = im_Married.transform(loan_data[['Married']]).ravel()

im_Dependents = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
im_Dependents = im_Dependents.fit(loan_data[['Dependents']])
loan_data['Dependents'] = im_Dependents.transform(loan_data[['Dependents']]).ravel()

im_Self_Employed = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
im_Self_Employed = im_Self_Employed.fit(loan_data[['Self_Employed']])
loan_data['Self_Employed'] = im_Self_Employed.transform(loan_data[['Self_Employed']]).ravel()

im_Edu = SimpleImputer(missing_values=np.NaN,strategy='most_frequent')
im_Edu = im_Edu.fit(loan_data[['Education']])
loan_data['Education']=im_Edu.transform(loan_data[['Education']]).ravel()

im_PA = SimpleImputer(missing_values=np.NaN,strategy='most_frequent')
im_PA = im_PA.fit(loan_data[['Property_Area']])
loan_data['Property_Area']=im_PA.transform(loan_data[['Property_Area']]).ravel()
logging.warning("Total waktu sampai proses imputasi data: "+str(datetime.now()-start_time))

# In[3]:


def data_prep(df):
    df.Gender = df.Gender.fillna('Others')
    df.Married = df.Married.fillna('Others')
    df.Dependents = df.Dependents.replace({'0':0,'1':1,'2':2,'3+':3}).fillna(0)
    df.Education = df.Education.replace({'Not Graduate':0,'Graduate':1})
    df.Self_Employed = df.Self_Employed.replace({'No':0,'Yes':1}).fillna(0)
    df.loc[df.ApplicantIncome>0,'ratio'] = ((df.LoanAmount/df.Loan_Amount_Term/30)/df.ApplicantIncome)
    df.loc[df.ApplicantIncome==0,'ratio'] = 0
    
    df.loc[(df.CoapplicantIncome>0), 'coratio'] = (df.LoanAmount/df.Loan_Amount_Term/30)/df.CoapplicantIncome
    df.loc[(df.CoapplicantIncome==0), 'coratio'] = 0
    df = df.drop(['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term'], axis=1)
    return df

df = data_prep(loan_data)
enc = OneHotEncoder(drop='first')
enc.fit(df.select_dtypes('object'))
x = enc.transform(df.select_dtypes('object')).toarray()
x = pd.DataFrame(x, columns=enc.get_feature_names(df.select_dtypes('object').columns))
df_train = pd.concat([df.select_dtypes(['int64','float64']),x],axis=1)
logging.warning("Total waktu sampai proses onehotencoding: "+str(datetime.now()-start_time))

# In[4]:


predictor_data = df_train.drop(['Loan_Status'],axis=1)
response_data = df_train['Loan_Status']

# In[6]:


xtrain,xtest,ytrain,ytest = train_test_split(predictor_data,response_data,test_size=0.2,random_state=4)
logging.warning("Total waktu sampai proses pemecahan data: "+str(datetime.now()-start_time))
xtrain_sc = scale(xtrain)
scaler = StandardScaler().fit(xtrain)
logging.warning("Total waktu sampai proses scaling data: "+str(datetime.now()-start_time))
model_loan = xgb.XGBClassifier(random_state=4,n_estimators=5,verbosity=3)
model_loan.fit(xtrain_sc, ytrain)

logging.warning("Total waktu sampai proses sebelum DASH: "+str(datetime.now()-start_time))

# In[7]:

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

app.layout = html.Div([
    # Baris 1
    dbc.Row([
        html.H1("Loan Prediction"),
        html.Br()
    ], justify='center'),

    # Baris 2
    dbc.Row([
        dbc.Col([

            html.H5('Married: '),
            dcc.Dropdown(id='married', value='No',
                         options=[{'label': 'No', 'value': 'No'},
                                  {'label': 'Yes', 'value': 'Yes'}],
                         ),
            html.H5('Dependents: '),
            dcc.Dropdown(id='dependents', value='1',
                         options=[{'label': '1', 'value': '1'},
                                  {'label': '2', 'value': '2'},
                                  {'label': '3+', 'value': '3+'}],
                         ),
            html.H5('Education: '),
            dcc.Dropdown(id='education', value='Graduate',
                         options=[{'label': 'Graduate', 'value': 'Graduate'},
                                  {'label': 'Not Graduate', 'value': 'Not Graduate'}],
                         ),
            html.H5('Self-Employed: '),
            dcc.Dropdown(id='employed', value='Yes',
                         options=[{'label': 'Yes', 'value': 'Yes'},
                                  {'label': 'No', 'value': 'No'}],
                         ),
            html.H5('LoanAmount: '),
            dbc.Input(id="loan", type="number", placeholder="",value=0),
        ], xs=6, sm=6, md=4, lg=2, xl=2),

        dbc.Col([
            html.H5('Gender:'),
            dcc.Dropdown(id='gender', value='Male',
                         options=[{'label': 'Male', 'value': 'Male'},
                                  {'label': 'Female', 'value': 'Female'}],
                         ),
            html.H5('Applicant Income:'),
            dbc.Input(id="app", type="number", placeholder="",value=0),
            html.H5('Coapplicant Income:'),
            dbc.Input(id="coapp", type="number", placeholder="",value=0),
            html.H5('Loan Amount Term:'),
            dbc.Input(id="term", type="number", placeholder="",value=0),
            html.H5('Credit History:'),
            dcc.Dropdown(id='credit', value=1,
                         options=[{'label': 1, 'value': 1},
                                  {'label': 0, 'value': 0}],
                         ),
            html.H5('Property_Area:'),
            dcc.Dropdown(id='area', value='Semiurban',
                         options=[{'label': 'Semiurban', 'value': 'Semiurban'},
                                  {'label': 'Urban', 'value': 'Urban'},
                                  {'label': 'Rural', 'value': 'Rural'}],
                         ),

        ], xs=6, sm=6, md=4, lg=2, xl=2)

    ], justify='center'),

    # Baris 3
    dbc.Row([
        dbc.Col([
            html.Br(),
            html.Center(dbc.Button('Predict', id='btn-nclicks-1', n_clicks=0,color="primary", className="mr-1",outline=True)),
            html.Center(html.H5('Loan Status:')),
            html.Center(html.H3(id='result2')),
        ], xs=5, sm=5, md=5, lg=5, xl=5),
    ], justify='center')
])


@app.callback(
    Output('result2', 'children'),
    [Input('gender', 'value'),
     Input('married', 'value'),
     Input('dependents', 'value'),
     Input('education', 'value'),
     Input('employed', 'value'),
     Input('credit', 'value'),
     Input('area', 'value'),
     Input('app', 'value'),
     Input('coapp', 'value'),
     Input('loan', 'value'),
     Input('term', 'value'),
     Input('btn-nclicks-1', 'n_clicks')
     ],
)
def label(gender, married, dependents, education, employed,credit, area, app, coapp,
          loan,term,btn1):
    
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    
    if 'btn-nclicks-1' in changed_id:
        x = pd.DataFrame({loan_data.columns[0]: gender,
                          loan_data.columns[1]: married,
                          loan_data.columns[2]: dependents,
                          loan_data.columns[3]: education,
                          loan_data.columns[4]: employed,
                          loan_data.columns[5]: app,
                          loan_data.columns[6]: coapp,
                          loan_data.columns[7]: loan,
                          loan_data.columns[8]: term,
                          loan_data.columns[9]: credit,
                          loan_data.columns[10]: area
                          }, index=[0])

        x_coba = data_prep(x)
        temp = enc.transform(x_coba.select_dtypes('object')).toarray()
        temp = pd.DataFrame(temp, columns=enc.get_feature_names(x_coba.select_dtypes('object').columns))
        temp = pd.concat([x_coba.select_dtypes(['int64','float64']),temp],axis=1)
        temp = scaler.transform(temp)
        temp = model_loan.predict(temp)

        if temp[0] == 1:
            return 'Granted'
        else:
            return 'Not Granted'


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)

