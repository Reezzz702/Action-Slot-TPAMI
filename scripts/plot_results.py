import matplotlib.pyplot as plt
import numpy as np

actor_table = ['c:z1-z2', 'c:z1-z3', 'c:z1-z4',
                'c:z2-z1', 'c:z2-z3', 'c:z2-z4',
                'c:z3-z1', 'c:z3-z2', 'c:z3-z4',
                'c:z4-z1', 'c:z4-z2', 'c:z4-z3',
                'c+:z1-z2', 'c+:z1-z3', 'c+:z1-z4',
                'c+:z2-z1', 'c+:z2-z3', 'c+:z2-z4',
                'c+:z3-z1', 'c+:z3-z2', 'c+:z3-z4',
                'c+:z4-z1', 'c+:z4-z2', 'c+:z4-z3',
                'b:z1-z2', 'b:z1-z3', 'b:z1-z4',
                'b:z2-z1', 'b:z2-z3', 'b:z2-z4',
                'b:z3-z1', 'b:z3-z2', 'b:z3-z4',
                'b:z4-z1', 'b:z4-z2', 'b:z4-z3',
                'b+:z1-z2', 'b+:z1-z3', 'b+:z1-z4',
                'b+:z2-z1', 'b+:z2-z3', 'b+:z2-z4',
                'b+:z3-z1', 'b+:z3-z2', 'b+:z3-z4',
                'b+:z4-z1', 'b+:z4-z2', 'b+:z4-z3',
                'p:c1-c2', 'p:c1-c4', 
                'p:c2-c1', 'p:c2-c3', 
                'p:c3-c2', 'p:c3-c4', 
                'p:c4-c1', 'p:c4-c3', 
                'p+:c1-c2', 'p+:c1-c4', 
                'p+:c2-c1', 'p+:c2-c3', 
                'p+:c3-c2', 'p+:c3-c4', 
                'p+:c4-c1', 'p+:c4-c3']

recall = [0.14461329, 0.12785457, 0.17672627, 0.11719304, 0.57140709, 0.23940338, 0.18105169, 0.16931818, 0.48541204, 0.31930411, 0.20417967, 0.36362695,
          0.17125254, 0.210399, 0.17760471, 0.23208267, 0.94256601, 0.14688703, 0.22741825, 0.19141469, 0.31740109, 0.21278386, 0.09905076, 0.5250832,
          0.24245111, 0.18912421, 0.13700543, 0.19114537, 0.48015036, 0.23432412, 0.11899559, 0.26907056, 0.31800636, 0.29123738, 0.2205487, 0.44683068,
          0.23576668, 0.17576557, 0.15181298, 0.04527398, 0.74300293, 0.25529534, 0.06147875, 0.27616533, 0.48553592, 0.33890506, 0.15116881, 0.14583713,
          0.23265702, 0.14946523, 0.3041035,  0.35672091, 0.43537108, 0.33659956, 0.10428006, 0.29660405,
          0.46650173, 0.00285618, 0.37712673, 0.56439982, 0.51681831, 0.56158509, 0.10657237, 0.15012201]
for i, r in enumerate(recall):
    recall[i] = np.round(r, 3)*100

ori_mAP = np.array([82.6, 91.2, 100, 98.7, 94.7, 87.8, 92.3, 99.2, 77.2, 89.6, 93.6, 92.4,
       93.8, 100, np.nan, 100, 96, 95.5, 95.6, 70, 97.3, 97.5, 100, 66.8,
       86.4, 95.8, 98.8, 89.6, 94.6, 100, 97, 90.3, 76.4, 97, 81.7, 100,
       98.3, 100, 100, 87.6, 91.4, 100, 90.7, 100, 100, 100, 50, 90.9,
       93.4, 83.2, 87.8, 85.7, 81.3, 84.5, 85.9, 83.9,
       95.8, 97, 91.9, 87.2, 95.6, 97.5, 89.6, 76.9])

mAP = np.array([82.9, 89.8, 98.4, 92.8, 76.2, 76.0, 86.0, 95.7, 37.1, 78.3, 88.9, 77.3,
       89.2, 100, np.nan, 100, 76.5, 94.6, 91.5, 70, 88.2, 95.2, 100, 46.9, 
       85.1, 89.6, 98.3, 79.6, 62.9, 86.1, 94.3, 55.5, 35.4, 65.8, 49, 85.3, 
       93.0, 100, 100, 87.6, 63.9, 46.7, 90.7, 87.0, 96.9, 100, 20, 74.2, 
       59.3, 73.7, 60.8, 53.9, 48.4, 24.5, 78.6, 41.0, 
       46.8, 97, 39.9, 51.9, 67.4, 27.0, 82.7, 53.2])

diff = ori_mAP - mAP
 
# plt.figure(figsize=(36, 18))
# plt.plot(actor_table, recall, label='Recall', color='royalblue', linewidth=4)
# plt.plot(actor_table, mAP, label='mAP', color='seagreen', linewidth=4)
# plt.plot(actor_table, diff, label='diff', color='goldenrod', linewidth=4)

# # Add labels and title
# plt.xlabel('Action', fontsize=18)
# plt.ylabel('Value', fontsize=18)
# plt.title('Recall and mAP', fontsize=22)
# plt.xticks(rotation=90, fontsize=16)
# plt.yticks(fontsize=16)
# plt.legend(fontsize=16)
# plt.grid(True)

# # Show the plot
# plt.show()
# plt.close

# Generate 64x64 matrix with values from 0 to 100
# confusion_matrix = np.random.randint(0, 101, (64, 64))
confusion_matrix = np.loadtxt('./results.txt', delimiter=',')

ori = np.stack([ori_mAP] * 64)
confusion_matrix = ori-confusion_matrix
# Create a single figure and axis
fig, ax = plt.subplots(figsize=(25, 20))
cmap = plt.get_cmap('coolwarm')
cax = ax.matshow(confusion_matrix, cmap=cmap)
fig.colorbar(cax)

# Add labels and title
ax.set_title('64x64 Confusion Matrix', fontsize=16)
ax.set_xlabel('Actions', fontsize=14)
ax.set_ylabel('Masked Actions', fontsize=14)
ax.set_xticks(np.arange(64))
ax.set_yticks(np.arange(64))
ax.set_xticklabels(actor_table, fontsize=12, rotation=90)
ax.set_yticklabels(actor_table, fontsize=12)

# Show the plot
plt.show()
