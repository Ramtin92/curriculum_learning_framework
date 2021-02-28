import numpy as np
from scipy import spatial
#test consine sim
# orthogonal
print("orthogonal:", spatial.distance.cosine([1, 0, 0], [0, 1, 0]))
# overlap (almost)
print("overlap (almost):", spatial.distance.cosine([0.1, 1, 0], [0, 1, 0]))

#diff
sim1 = np.load("diff1.npy", allow_pickle=True).tolist()
sim2 = np.load("diff2.npy", allow_pickle=True).tolist()
sim1_W1 = np.stack([elem['W1']for elem in sim1])
sim1_W1.resize((1001,370))
sim1_W2 = np.stack([elem['W2']for elem in sim1])
sim1_W2.resize((1001,50))
sim2_W1 = np.stack([elem['W1']for elem in sim2])
sim2_W1.resize((1001,370))
sim2_W2 = np.stack([elem['W2']for elem in sim2])
sim2_W2.resize((1001,50))

sim1_W1_mean = np.mean(sim1_W1,axis=0)
sim1_W2_mean = np.mean(sim1_W2,axis=0)
sim2_W1_mean = np.mean(sim2_W1,axis=0)
sim2_W2_mean = np.mean(sim2_W2,axis=0)
W1_sim = spatial.distance.cosine(sim1_W1_mean, sim2_W1_mean)
W2_sim = spatial.distance.cosine(sim1_W2_mean, sim2_W2_mean)
print("diff W1:", W1_sim)
print("diff W2:", W2_sim)

#sim
sim1 = np.load("sim1.npy", allow_pickle=True).tolist()
sim2 = np.load("sim2.npy", allow_pickle=True).tolist()
sim1_W1 = np.stack([elem['W1']for elem in sim1])
sim1_W1.resize((1001,370))
sim1_W2 = np.stack([elem['W2']for elem in sim1])
sim1_W2.resize((1001,50))
sim2_W1 = np.stack([elem['W1']for elem in sim2])
sim2_W1.resize((1001,370))
sim2_W2 = np.stack([elem['W2']for elem in sim2])
sim2_W2.resize((1001,50))

sim1_W1_mean = np.mean(sim1_W1,axis=0)
sim1_W2_mean = np.mean(sim1_W2,axis=0)
sim2_W1_mean = np.mean(sim2_W1,axis=0)
sim2_W2_mean = np.mean(sim2_W2,axis=0)
W1_sim = spatial.distance.cosine(sim1_W1_mean, sim2_W1_mean)
W2_sim = spatial.distance.cosine(sim1_W2_mean, sim2_W2_mean)

print("sim W1:", W1_sim)
print("sim W2:", W2_sim)