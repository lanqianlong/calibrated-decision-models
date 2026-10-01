import matplotlib.pyplot as plt
import numpy as np

def write_reliability_diagram(labels, probabilities, path, bins=10):
    edges = np.linspace(0,1,bins+1); xs=[]; ys=[]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lo) & (probabilities <= hi if hi == 1 else probabilities < hi)
        if np.any(mask): xs.append(float(np.mean(probabilities[mask]))); ys.append(float(np.mean(labels[mask])))
    fig, ax = plt.subplots(); ax.plot([0,1],[0,1],'--',label='perfect'); ax.plot(xs,ys,'o-',label='model'); ax.set(xlabel='Mean predicted probability', ylabel='Observed frequency'); ax.legend(); fig.savefig(path, dpi=160, bbox_inches='tight'); plt.close(fig); return path

