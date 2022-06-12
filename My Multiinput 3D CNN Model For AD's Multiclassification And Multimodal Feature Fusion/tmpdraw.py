import pandas as pd
import matplotlib.pyplot as plt

f_t = pd.read_csv(r"C:\Users\lj\Desktop\毕设\MyResults\Logs\b1_dt_logs\run-train-tag-epoch_loss.csv")
f_v = pd.read_csv(r"C:\Users\lj\Desktop\毕设\MyResults\Logs\b1_dt_logs\run-validation-tag-epoch_loss.csv")

df_t = pd.DataFrame(f_t)
df_v = pd.DataFrame(f_v)

dt_t = df_t['Value']
dt_v = df_v['Value']
epochs = range(1, len(dt_t)+1)
metric = 'loss'
plt.plot(epochs, dt_t, 'bo--')
plt.plot(epochs, dt_v, 'ro-')
plt.title('Training and validation ' + metric)
plt.xlabel("Epochs")
plt.ylabel(metric)
plt.legend(["train_" + metric, 'val_' + metric])
plt.ylim(0, 15)
savefigpath = r'C:\Users\lj\Desktop\毕设\MyResults\ResultsFig\b1_dt_loss.png'
plt.savefig(savefigpath)
plt.show()
