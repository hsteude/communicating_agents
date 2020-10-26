import plotly.graph_objects as go


def plot_learning_curve(epoch_ls, train_loss_ls, val_loss_ls, path):
    """Plots training and validation loss over epochs"""

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epoch_ls,
                             y=[t.detach().cpu().numpy() for t in train_loss_ls],
                             mode='lines',
                             name='training loss'))
    fig.add_trace(go.Scatter(x=epoch_ls,
                             y=[t.detach().cpu().numpy() for t in val_loss_ls],
                             mode='lines', name='validation loss'))
    fig.update_layout(
        title="Learning curve",
        xaxis_title="Epochs",
        yaxis_title="Loss")
    fig.write_html(path)
