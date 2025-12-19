import plotly.graph_objects as go
import pandas as pd


def graphic(data: dict):
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
    return fig
