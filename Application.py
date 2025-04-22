import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import tensorflow as tf
from dash.dependencies import Input, Output, State
import joblib

# Load saved assets
scaler = joblib.load("scaler_full.pkl") 

label_encoder = joblib.load("label_encoder.pkl")
model = tf.keras.models.load_model("gradeclass_model.keras")  

# Match feature names the scaler was trained on
feature_names = [
    'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly', 'Absences',
    'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports'
]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container([
    html.Br(),
    dbc.Card([
        dbc.CardBody([
            html.H2("BrightPath Grade Predictor", className="text-center text-primary mb-4"),

            dbc.Row([
                dbc.Col([
                    dbc.Label("Ethnicity"),
                    dcc.Dropdown(
                        id='ethnicity',
                        options=[
                            {'label': 'Caucasian', 'value': 0},
                            {'label': 'African American', 'value': 1},
                            {'label': 'Asian', 'value': 2},
                            {'label': 'Other', 'value': 3},
                        ],
                        placeholder='Select...',
                        className="form-control"
                    )
                ], md=4),

                dbc.Col([
                    dbc.Label("Parental Education"),
                    dcc.Dropdown(
                        id='parentaledu',
                        options=[
                            {'label': 'None', 'value': 0},
                            {'label': 'High School', 'value': 1},
                            {'label': 'Some College', 'value': 2},
                            {'label': "Bachelor's", 'value': 3},
                            {'label': 'Higher Study', 'value': 4},
                        ],
                        placeholder='Select...',
                        className="form-control"
                    )
                ], md=4),

                dbc.Col([
                    dbc.Label("Study Time Weekly"),
                    dcc.Input(id='study-time', type='number', placeholder='e.g. 10', className="form-control")
                ], md=4),

                dbc.Col([
                    dbc.Label("Absences"),
                    dcc.Input(id='absences', type='number', placeholder='e.g. 3', className="form-control")
                ], md=4),

                dbc.Col([
                    dbc.Label("Tutoring"),
                    dcc.Dropdown(
                        id='tutoring',
                        options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
                        placeholder='Select...',
                        className="form-control"
                    )
                ], md=4),

                dbc.Col([
                    dbc.Label("Parental Support"),
                    dcc.Dropdown(
                        id='parental-support',
                        options=[
                            {'label': 'None', 'value': 0},
                            {'label': 'Low', 'value': 1},
                            {'label': 'Moderate', 'value': 2},
                            {'label': 'High', 'value': 3},
                            {'label': 'Very High', 'value': 4},
                        ],
                        placeholder='Select...',
                        className="form-control"
                    )
                ], md=4),

                dbc.Col([
                    dbc.Label("Extracurricular"),
                    dcc.Dropdown(
                        id='extracurricular',
                        options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
                        placeholder='Select...',
                        className="form-control"
                    )
                ], md=4),

                dbc.Col([
                    dbc.Label("Sports"),
                    dcc.Dropdown(
                        id='sports',
                        options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
                        placeholder='Select...',
                        className="form-control"
                    )
                ], md=4),
            ], className="mb-4"),

            dbc.Button("Predict Grade", id='submit-button', color="success", className="w-100 mb-3", n_clicks=0),
            dbc.Spinner(html.Div(id='prediction-output', className="text-center text-info"), size="lg", color="primary")
        ])
    ], className="shadow-lg p-4 bg-white rounded")
], fluid=True)

@app.callback(
    Output('prediction-output', 'children'),
    Input('submit-button', 'n_clicks'),
    State('ethnicity', 'value'),
    State('parentaledu', 'value'),
    State('study-time', 'value'),
    State('absences', 'value'),
    State('tutoring', 'value'),
    State('parental-support', 'value'),
    State('extracurricular', 'value'),
    State('sports', 'value')
)
def predict(n_clicks, ethnicity, parentaledu, study_time, absences, tutoring,
            parental_support, extracurricular, sports):

    if n_clicks > 0:
        if None in (ethnicity, parentaledu, study_time, absences, tutoring,
                    parental_support, extracurricular, sports):
            return "Please fill in all fields."

        try:
            # Create DataFrame with correct column names
            input_df = pd.DataFrame([[
                ethnicity, parentaledu, study_time, absences,
                tutoring, parental_support, extracurricular, sports
            ]], columns=feature_names)

            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
            predicted_class_index = np.argmax(prediction)
            predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]

            return f"Predicted Grade: {predicted_label}"

        except Exception as e:
            return f"Error during prediction: {str(e)}"

    return ""

if __name__ == '__main__':
    app.run(debug=True)
