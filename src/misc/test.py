fnirs = [445,
         1169,
         1895,
         2619,
         3341,
         4064,
         4788,
         5513,
         6238,
         7794,
         8519,
         9242,
         9966,
        10690,
        11415,
        12138,
        12863,
        13587,
        15169,
        15894,
        16617,
        17342,
        18065,
        18790,
        19515,
        20239,
        20960]

eeg =   [46968,
        116495,
        185982,
        255507,
        324916,
        394528,
        463974,
        533587,
        603078,
        755221,
        824795,
        894432,
        963931,
        1033256,
        1102683,
        1172243,
        1241840,
        1311388,
        1461180,
        1530761,
        1600188,
        1669726,
        1739167,
        1808726,
        1878356,
        1947860,
        2017105]

fnirs_time = [42.68,
                112.26,
                181.90,
                251.40,
                320.71,
                390.13,
                459.69,
                529.29,
                598.83]

eeg_time = [46968,
        116495,
        185982,
        255507,
        324916,
        394528,
        463974,
        533587,
        603078]

sub_fnirs = [445,
         1169,
         1895,
         2619,
         3341,
         4064,
         4788,
         5513,
         6238]

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import linregress

# m is ratio
# c is constant offset start

fnirs_time = np.array(fnirs_time)
eeg_time = np.array(eeg_time)/1000

lr = linregress(eeg_time, fnirs_time)
print(lr)

m_1=lr.slope
c_1=lr.intercept

sub_fnirs = np.array(sub_fnirs)

lr = linregress(fnirs_time, sub_fnirs)
print(lr)

m_2=lr.slope
c_2=lr.intercept

test = []
for i in eeg:
    i = i/1000
    transformed_eeg = ((i*m_1+c_1)*m_2+c_2)
    test.append(int(np.rint(transformed_eeg)))

print(test)
print(fnirs)

# eeg_time to fnirs_time
#slope=0.9998271160946597, intercept=-4.182145391771428
# fnirs_time to fnirs
#slope=10.416067241966067, intercept=0.2177377600651198