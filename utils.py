from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_roc(tpr, fpr, savedir=None):
    data = pd.DataFrame(dict(tpr=tpr, fpr=fpr))
    p = sns.lineplot(data=data, x='fpr', y='tpr')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    fig = p.get_figure()
    if savedir is not None:
        fig.savefig(Path(savedir) / 'roc.png')
    plt.close(fig)
    return fig

