import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_roc(tpr, fpr):
    data = pd.DataFrame(dict(tpr=tpr, fpr=fpr))
    p = sns.lineplot(data=data, x='fpr', y='tpr')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    fig = p.get_figure()
    plt.close(fig)
    return fig


