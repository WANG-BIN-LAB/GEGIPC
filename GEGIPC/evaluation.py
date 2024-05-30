import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def L1error(pred, true):
    return np.mean(np.abs(pred - true))

def CCCscore(y_pred, y_true):
    # pred: shape{n sample, m cell}
    ccc_value = 0
    temp = []
    for i in range(y_pred.shape[1]):
        r = np.corrcoef(y_pred[:, i], y_true[:, i])[0, 1]
        # print(r)
        # Mean
        mean_true = np.mean(y_true[:, i])
        mean_pred = np.mean(y_pred[:, i])
        # Variance
        var_true = np.var(y_true[:, i])
        var_pred = np.var(y_pred[:, i])
        # Standard deviation
        sd_true = np.std(y_true[:, i])
        sd_pred = np.std(y_pred[:, i])
        # Calculate CCC
        numerator = 2 * r * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        ccc = numerator / denominator
        # print(ccc)
        ccc_value += ccc
        temp.append(ccc)
    return ccc_value / y_pred.shape[1]

def RMSEscore(pred, true):
    return np.sqrt(np.mean((pred - true)**2))


def score(pred, label):
    ccc_mean = CCCscore(pred, label)
    new_pred = pred.reshape(pred.shape[0] * pred.shape[1], 1)
    new_label = label.reshape(label.shape[0] * label.shape[1], 1)
    rmse = RMSEscore(new_pred, new_label)
    mae = L1error(new_pred, new_label)
    ccc = CCCscore(new_pred, new_label)
    return mae,rmse,ccc,ccc_mean

def showloss(loss):
    plt.figure()
    plt.plot(loss)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()

def showheatmap(pred,real):
    #
    fig, ax = plt.subplots(1, 3, constrained_layout=True, figsize=(16, 4.6))

    corr = sns.heatmap(data=pred,
                       vmin=0,
                       vmax=0.5,
                       cmap='bwr',
                       ax=ax[0])
    corr.set_title('pred')
    corr1 = sns.heatmap(data=real,
                       vmin=0,
                       vmax=0.5,
                       cmap='bwr',
                       ax=ax[1])
    corr1.set_title('test_y')
    corr1 = sns.heatmap(data=pred-real,
                       vmin=-0.5,
                       vmax=0.5,
                       cmap='bwr',
                       ax=ax[2])
    corr1.set_title('dist')
    plt.show()