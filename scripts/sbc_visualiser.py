from functools import cache
from glob import glob
from dash import Dash, html, dcc, Input, Output, callback
from tcup_paper.data.io import load_dataset
import numpy as np
import plotly.subplots as ps
import plotly.graph_objects as go
import scipy.stats as sps
import xarray as xr

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

dataset_type = [x.split("/")[-1] for x in glob("data/sbc/*")]
dataset_type.sort()
file_index = list(range(1, 401))

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label="SBC rank histograms", children=[
            html.Div([
                dcc.Dropdown(
                    dataset_type,
                    "t",
                    id='rank-data-filter',
                ),
                dcc.Checklist(
                    ["tcup", "ncup", "tobs", "fixed3"],
                    ["tcup"],
                    id='rank-model-filter',
                    inline=True,
                ),
            ]),
            html.Div([
                dcc.Graph(
                    id='rank-plot',
                )
            ], style={'width': '99%', 'display': 'inline-block', 'padding': '0 20'}),
        ]),
        dcc.Tab(label="SBC datasets", children=[
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        dataset_type,
                        "outlier10",
                        id='data-filter',
                    ),
                    dcc.Dropdown(
                        file_index,
                        1,
                        id='file-filter',
                    ),
                    dcc.Checklist(
                        ["tcup", "ncup", "tobs", "fixed3"],
                        ["tcup"],
                        id='model-filter',
                        inline=True,
                    ),
                    dcc.Checklist(
                        {True: "Show posterior predictive"},
                        id='show-post-pred',
                        inline=True,
                    ),
                ],
                style={'width': '49%', 'display': 'inline-block'}),

                html.Div([
                ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
            ], style={
                'padding': '10px 5px'
            }),

            html.Div([
                dcc.Graph(
                    id='dataset-plot',
                )
            ], style={'width': '99%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([
            ], style={'display': 'inline-block', 'width': '49%'}),
                ]),
            ]),

])

def pool_bins(bins, pooling_factor):
    # Check pooling factor, bin sizes are powers of 2
    assert np.isclose(
        np.mod(np.log2(pooling_factor), 1), 0
    ), "pooling_factor must be power of 2"
    assert np.isclose(
        np.mod(np.log2(bins.shape[0]), 1), 0
    ), "len(bins) must be power of 2"

    if pooling_factor == 1:
        return bins
    else:
        new_bins = np.zeros(bins.shape[0] // 2)
        new_bins = bins[:-1:2] + bins[1::2]
        return pool_bins(new_bins, pooling_factor // 2)


@cache
def get_binned_ranks(dataset, model):
    results_path = f"results/sbc/{model}/{dataset}/"
    data_path = f"data/sbc/{dataset}/"

    N = 0
    L = 1023
    B = 16

    var_names = None

    for filename in glob(results_path + "*.nc"):
        # Attempt to load corresponding data file
        data_filename = (
            data_path + filename.split("/")[-1].split(".")[0] + ".json"
        )
        try:
            _, info = load_dataset(data_filename)
        except:
            continue

        sbc_data = xr.load_dataset(filename)

        if var_names is None:
            var_names = []
            for x in sbc_data.keys():
                if "rank_" in x:
                    continue
                elif "true_" in x:
                    continue
                elif "post_pred_" in x:
                    continue
                elif x == "beta_scaled":
                    for idx in range(sbc_data.sizes["beta_scaled_dim_0"]):
                        var_names.append(f"beta_scaled.{idx}")
                else:
                    var_names.append(x)

            ranks = {var_name: [] for var_name in var_names}
            bins = np.zeros((len(var_names), L + 1))

        for var_name, curr_bins in zip(var_names, bins):
            if "beta_scaled" in var_name:
                idx = int(var_name.split(".")[-1])
                # Save rank and dataset value to appropriate value
                ranks[var_name].append((
                    sbc_data["true_beta_scaled"].values[idx],
                    sbc_data["rank_beta_scaled"].values[idx],
                    sbc_data["beta_scaled"].median(axis=-1)[idx],
                ))
                bin_idx = int(sbc_data["rank_beta_scaled"].values[idx] * L)
                curr_bins[bin_idx] += 1
            else:
                # Save rank and dataset value to appropriate value
                ranks[var_name].append((
                    sbc_data[f"true_{var_name}"].values[()],
                    sbc_data[f"rank_{var_name}"].values[()],
                    sbc_data[var_name].median(axis=-1)[()],
                ))
                bin_idx = int(sbc_data[f"rank_{var_name}"].values[()] * L)
                curr_bins[bin_idx] += 1

        # Increment number of datasets
        N += 1

    return var_names, bins


@callback(
    Output('rank-plot', 'figure'),
    Input('rank-data-filter', 'value'),
    Input('rank-model-filter', 'value'),
)
def update_rank_hist(dataset_name, model_names):
    if dataset_name in ["t", "tobs", "fixed"]:
        n_plots = 4
    else:
        n_plots = 3

    fig = ps.make_subplots(rows=1, cols=n_plots)

    L = 1023
    B = 16

    for model in model_names:
        if model == "tcup":
            model_color = "red"
        elif model == "ncup":
            model_color = "blue"
        elif model == "tobs":
            model_color = "green"
        elif model == "fixed3":
            model_color = "orange"

        var_names, bins = get_binned_ranks(dataset_name, model)
        N = bins[0, :].sum()

        confidence_levels = [1 - 1 / B, 1 - 1 / (len(var_names) * B)]
        confidence_intervals = []
        for confidence_level in confidence_levels:
            cdf_vals = [0.5 - confidence_level / 2, 0.5 + confidence_level / 2]
            lower, upper = sps.binom.ppf(cdf_vals, N, 1 / B)
            confidence_intervals.append((lower / N, upper / N))
            print(f"{N=}, {B=}, {N / B=:.2f}")
            print(
                f"{confidence_level * 100:.0f}% confidence interval: [{lower}, {upper}]"
            )

        for idx, (var_name, curr_bins) in enumerate(zip(var_names, bins)):
            pooled_bins = pool_bins(curr_bins, pooling_factor=(L + 1) // B)
            fig.add_trace(
                go.Bar(
                    x0=0,
                    dx=1 / B,
                    y=pooled_bins / N,
                    offset=0,
                    hoverinfo="none",
                    width=1 / B,
                    name=model,
                    opacity=0.5,
                    marker={"color": model_color},
                    showlegend=(idx == 0)
                ),
                row=1, col=idx + 1,
            )

    fig.add_hline(y=1 / B, line_dash="dash")
    for lower, upper in confidence_intervals:
        fig.add_hrect(y0=lower, y1=upper, line_width=0, fillcolor="black", opacity=0.1)

    return fig

def add_model_samples(fig, model_name, dataset_name, file_name, data, info, show_post_pred):
    try:
        samples = xr.open_dataset(f"results/sbc/{model_name}/{dataset_name}/{file_name}.nc")
    except FileNotFoundError:
        return

    line_style = {}
    if model_name == "tcup":
        line_style["color"] = "red"
    elif model_name == "ncup":
        line_style["color"] = "blue"
    elif model_name == "tobs":
        line_style["color"] = "green"
    elif model_name == "fixed3":
        line_style["color"] = "orange"

    x_true = np.sort(info["true_x"], axis=0)
    for idx in range(100):
        y_true = np.array(samples["alpha_scaled"][idx]) + np.dot(x_true, np.array(samples["beta_scaled"][0, idx]))
        fig.add_trace(go.Scatter(
            x=x_true.flatten(),
            y=y_true.flatten(),
            line=line_style,
            opacity=0.02,
            mode="lines",
            name=f"{model_name}{idx}",
            showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=[0],
        y=[[0]],
        line=line_style,
        opacity=1,
        mode="lines",
        name=model_name,
    ))

    if show_post_pred:
        for idx in range(100):
            fig.add_trace(go.Scatter(
                x=samples["post_pred_x_scaled"][:, 0, idx],
                y=samples["post_pred_y_scaled"][:, idx],
                # error_x=dict(
                #     type='data',
                #     array=np.sqrt(data["cov_x_scaled"]).flatten(),
                #     visible=True
                # ),
                # error_y=dict(
                #     type='data',
                #     array=np.array(data["dy_scaled"]),
                #     visible=True,
                # ),
                mode='markers',
                marker_color='red',
                name=f"Post. pred.{idx}",
            ))


@callback(
    Output('dataset-plot', 'figure'),
    Input('data-filter', 'value'),
    Input('file-filter', 'value'),
    Input('model-filter', 'value'),
    Input('show-post-pred', 'value'),
)
def update_graph(dataset_name, file_name, model_names, show_post_pred):
    data, info = load_dataset(f"data/sbc/{dataset_name}/{file_name}.json")

    x = np.array(data["x_scaled"])
    y = np.array(data["y_scaled"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x[:, 0],
        y=y,
        error_x=dict(
            type='data',
            array=np.sqrt(data["cov_x_scaled"]).flatten(),
            visible=True
        ),
        error_y=dict(
            type='data',
            array=np.array(data["dy_scaled"]),
            visible=True,
        ),
        mode='markers',
        marker_color='rgba(0, 0, 0, 0.8)',
        name="Data",
    ))

    x_true = np.sort(info["true_x"], axis=0)

    for model_name in model_names:
        add_model_samples(fig, model_name, dataset_name, file_name, data, info, show_post_pred)

    y_true = np.array(info["alpha_scaled"]) + np.dot(x_true, np.array(info["beta_scaled"]))
    fig.add_trace(go.Scatter(
        x=x_true.flatten(),
        y=y_true,
        line=dict(color="black", dash="dash"),
        opacity=1,
        mode="lines",
        name="Intrinsic",
    ))

    return fig


if __name__ == '__main__':
    app.run(debug=True)

