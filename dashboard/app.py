import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd

# Load predictions
preds = pd.read_parquet("/Users/murad/nasa_proj/data/processed/baseline_predictions_FD001.parquet")

# Load raw test data properly (IMPORTANT FIX)
cols = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f'sensor_{i}' for i in range(1, 22)]
data = pd.read_csv(
    "/Users/murad/nasa_proj/data/raw/test_FD001.txt",
    sep=r"\s+",
    header=None
)
data.columns = cols

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Fleet Monitoring Dashboard", style={"textAlign": "center"}),

    html.Div([
        dcc.Dropdown(
            id="engine-dropdown",
            options=[{"label": f"Engine {i}", "value": i} for i in data["unit"].unique()],
            value=data["unit"].unique()[0],
            style={"width": "45%"}
        ),

        dcc.Dropdown(
            id="sensor-dropdown",
            options=[{"label": f"Sensor {i}", "value": f"sensor_{i}"} for i in range(1, 22)],
            value="sensor_1",
            style={"width": "45%"}
        )
    ], style={"display": "flex", "justifyContent": "space-between"}),

    dcc.Graph(id="sensor-plot"),

    html.Div([
        dcc.Graph(id="rul-gauge", style={"width": "50%"}),
        html.Div(id="rul-output", style={"fontSize": 24, "padding": "20px"})
    ], style={"display": "flex"})
])


@app.callback(
    [Output("sensor-plot", "figure"),
     Output("rul-output", "children"),
     Output("rul-gauge", "figure")],
    [Input("engine-dropdown", "value"),
     Input("sensor-dropdown", "value")]
)
def update_dashboard(engine_id, sensor):
    df_engine = data[data["unit"] == engine_id]

    # 📈 Sensor plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_engine["cycle"],
        y=df_engine[sensor],
        mode="lines",
        name=sensor
    ))
    fig.update_layout(title=f"{sensor} over time")

    # 🔢 Prediction
    pred_row = preds[preds["unit"] == engine_id]

    if pred_row.empty:
        return fig, "No prediction available", go.Figure()

    pred = int(pred_row.iloc[0]["xgb_RUL"])

    # 🚦 Risk logic
    if pred < 20:
        color = "red"
        status = "HIGH RISK"
    elif pred < 50:
        color = "orange"
        status = "MEDIUM RISK"
    else:
        color = "green"
        status = "LOW RISK"

    # 📊 Gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        title={'text': "Remaining Useful Life"},
        gauge={
            'axis': {'range': [0, 150]},
            'bar': {'color': color}
        }
    ))

    text = f"Predicted RUL: {pred} cycles | Status: {status}"

    return fig, text, gauge


if __name__ == "__main__":
    app.run(debug=True)