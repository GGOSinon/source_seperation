from AI_seperation import Model
import numpy as np

data = np.load("../lib/data.npy")
num_data = data.shape[0]
bass, drums, other, vocals, mixture = [], [], [], [], []
for i in range(num_data):
	bass.append(data[i][0])
	drums.append(data[i][1])
	other.append(data[i][2])
	vocals.append(data[i][3])
	mixture.append(data[i][4])
		
trainData = [mixture, vocals]
model = Model(trainData)
max_step = 10000

avg_loss, avg_regloss = 0, 0
DISPLAY_STEP = 10

for step in range(1, max_step+1):
	loss, regloss, learning_rate = model.train()
	avg_loss += loss/DISPLAY_STEP
	avg_regloss += regloss/DISPLAY_STEP
	if step % DISPLAY_STEP == 0:
		print("Step %d - loss : %.5f, reg_loss : %.5f" % (step, avg_loss, avg_regloss))
		avg_loss, avg_regloss = 0, 0
