import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline
import math
from scipy.interpolate import interp1d

fname_l1_2 = 'segan_v2/g_l1_losses.txt'
fname_adv_2 = 'segan_v2/g_adv_losses.txt'

fname_l1_3 = 'segan_v4/g_l1_losses.txt'
fname_adv_3 = 'segan_v4/g_adv_losses.txt'

with open(fname_l1_2) as f:
    content_l1_2= f.readlines()

with open(fname_adv_2) as f:
    content_adv_2 = f.readlines()

with open(fname_l1_3) as f:
    content_l1_3= f.readlines()

with open(fname_adv_3) as f:
    content_adv_3 = f.readlines()

# you may also want to remove whitespace characters like `\n` at the end of each line

content_l1_2 = [x.strip() for x in content_l1_2]
content_l1_2 = [float(x)/100. for x in content_l1_2]
content_adv_2 = [x.strip() for x in content_adv_2]
content_adv_2 = [float(x) for x in content_adv_2]

content_l1_3 = [x.strip() for x in content_l1_3]
content_l1_3 = [float(x)/100. for x in content_l1_3]
content_adv_3 = [x.strip() for x in content_adv_3]
content_adv_3 = [float(x) for x in content_adv_3]


res_l1_2 = []
res_adv_2 = []
res_l1_3 = []
res_adv_3 = []
step_2 = 3
step_3 = 3
for i in range(len(content_l1_2)):
    if i%3==0 and i<len(content_l1_2)-step_2:
        res_l1_2.append(sum(content_l1_2[i:i+step_2])/step_2)
        res_adv_2.append(sum(content_adv_2[i:i+step_2])/step_2)

for i in range(len(content_l1_3)):
    if i<len(content_l1_3)-step_3:
        res_l1_3.append(sum(content_l1_3[i:i+step_3])/step_3)
        res_adv_3.append(sum(content_adv_3[i:i+step_3])/step_3)
x = np.linspace(0, len(res_l1_3)-1, num=len(res_l1_3), endpoint=True)
f1_l1_3 = interp1d(x, np.asarray(res_l1_3), kind='linear')
f2_l1_3 = interp1d(x, np.asarray(res_l1_3), kind='linear')
f1_adv_3 = interp1d(x, np.asarray(res_adv_3), kind='nearest')
f2_adv_3 = interp1d(x, np.asarray(res_adv_3), kind='nearest')
out_l1_3 = []
out_adv_3 = []
for i in range(len(res_l1_3)):
    out_l1_3.append(res_l1_3[i])
    out_l1_3.append(f1_l1_3(x[i]))
    out_adv_3.append(res_adv_3[i])
    out_adv_3.append(f1_adv_3(x[i]))

    if i%2==0:
        out_l1_3.append(f2_l1_3(x[i]))
        out_adv_3.append(f2_adv_3(x[i]))


fig = plt.figure()

plt.subplot(2, 1, 1)
plt.plot(res_l1_2,label='FSEGAN')
plt.plot(out_l1_3, label='improved FSEGAN')
plt.xlabel("batch number")
plt.ylabel('generator l1 loss')
plt.title("enhanced cepstrum v.s. clean cepstrum")
plt.legend()
plt.grid()
plt.tight_layout(pad=3, w_pad=1, h_pad=2.0)
plt.subplot(2, 1, 2)
plt.xlabel("batch number")
plt.ylabel("generator adversarial loss")
plt.title("adversarial loss")
plt.plot(res_adv_2, label='FSEGAN')
plt.plot(out_adv_3, label='improved FSEGAN')
plt.legend()

# plt.subplot(2, 2, 3)
# plt.plot(res_fk)
#
# plt.subplot(2, 2, 4)
# plt.plot(res_rl)
plt.grid()
plt.savefig("loss_comparison.png")
plt.show()