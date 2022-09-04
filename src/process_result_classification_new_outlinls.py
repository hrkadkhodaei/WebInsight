import matplotlib.pyplot as plt
from collections import OrderedDict
import os

linestyles_dict = OrderedDict(
    [('solid', (0, ())),
     ('dotted', (0, (1, 2))),
     ('densely dotted', (0, (1, 1))),
     ('dashed', (0, (3, 2))),
     ('densely dashed', (0, (3, 2))),
     ('dashdotted', (0, (3, 1, 1, 1))),
     ('double dashdotted', (0, (3, 1, 1, 1, 1, 1)))
     ]
)
colours = {
    'Recall': "black",
    'Precision': "black",
    'F1-score': "orange",
    'Accuracy-(b)': "green"
}
linestyle = {
    'Recall': "dashdotted",
    'Precision': "dashed",
    'F1-score': "solid",
    'Accuracy-(b)': "solid"
}
baseline_score = {
    "Internal": {
        'Recall': 0.7626987960760999,
        'Precision': 0.7574287111266917,
        'F1-score': 0.7600546182508273,
        'Accuracy-(b)': 0.8338315521900334
    },
    "External": {
        'Recall': 0.5881507341616182,
        'Precision': 0.6017438806597332,
        'F1-score': 0.5948696645549901,
        'Accuracy-(b)': 0.7780618387258953
    }
}

which_model = 'LGB'
which_target = 'External'
target = 'diff' + which_target + 'OutLinks'
which_features = ['SP', 'SN', 'DN8', 'DP8', 'DPRate']
title = "target " + target + \
        "-features " + '_'.join(which_features) + \
        "-model " + which_model

path = r'result/'
fig_path = 'figures-newOutlinks/'
file_name = 'Classification_newOutlink_External_SP_SN_DP8_DPRate_ET.log'
file_name = 'classification_newOutlink_SP_SN_DP8_DN8_DPRate_Internal_ET.log'
file_name = 'Classification_SP_SN_DP8_DN8_DPRate_ET.log'
file_name = 'classification_newOutlink_SP_SN_DP8_DN8_DPRate_' + which_target + '_ET.log'
file_name = 'lgb_100_200_300.log'
file_name = 'Classification_newOutlink_SP_SN_DP8_DN8_DPRate_External_LGB.log'

dic1 = {
    'DP0': {'Accuracy-(b)': 0.0, 'Recall': 0.0, 'Precision': 0.0, 'F1-score': 0.0},
}
for i in range(1, 9):
    dic1.update({'DP' + str(i): {'Accuracy-(b)': 0.0, 'Recall': 0.0, 'Precision': 0.0, 'F1-score': 0.0}})

if not os.path.isdir(fig_path):
    os.mkdir(fig_path)
with open(path + file_name) as f:
    lines = f.readlines()
    block = -1
    for line in lines:
        if line.startswith('diff' + which_target + 'OutLinks  -->  DP'):
            block += 1
        elif line.startswith('\t\tTest'):
            words = line.split()
            if len(words) > 3:
                words[1] = '-'.join(words[1:3])
            dic1['DP' + str(block)][words[1]] += float(words[-1]) / 3.0
        # else:
        #     continue

dic11 = {'Accuracy-(b)': [], 'Recall': [], 'Precision': [], 'F1-score': []}
for key in dic1.keys():
    for key2 in dic1[key]:
        dic11[key2].append(dic1[key][key2])

fig, ax = plt.subplots(figsize=(3, 3.6))

x = range(9)
for s in dic11.keys():
    # averaged_scores = [np.mean(e) for e in scores[s]]
    line, = ax.plot(x, dic11[s],
                    linestyle=linestyles_dict[linestyle[s]], color=colours[s], linewidth=2, label=s.replace('-', ' '))
for s in ['F1-score']:
    line, = ax.plot(x[1:], [baseline_score[which_target][s]] * (len(x) - 1),
                    linestyle=linestyles_dict["densely dotted"], color="grey", linewidth=2,
                    label='(dynamic baseline\n ' + s.replace('-', ' ') + ')')

ax.yaxis.set_ticks_position('both')
plt.xticks(x, x)
plt.ylim((.29, .95))
plt.xlabel('history size')
plt.ylabel('performance')
ax.legend(loc='lower right', frameon=False, fontsize=9, labelspacing=0.2)

plt.tight_layout()
# plt.show()

plt.savefig(path + fig_path + "all_scores-" + title + ".png", dpi=500, bbox_inches='tight')
plt.close()

print(dic1)
