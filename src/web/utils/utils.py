import plotly.graph_objects as go
import pandas as pd


def candle_graphic(data: dict):
    df = pd.DataFrame(data)
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["datetime"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
            )
        ]
    )
    # fig.update_layout(title = "Candlestick Chart")
    fig.update_xaxes(title_text="data")
    fig.update_yaxes(title_text="price")
    return fig


def line_graphic(data: dict):
    df = pd.DataFrame(data)
    fig = go.Figure(data=[go.Line(x=df["dt"], y=df["price"])])
    # fig.update_layout(title = "Test predict")
    fig.update_xaxes(title_text="data")
    fig.update_yaxes(title_text="price")
    return fig
