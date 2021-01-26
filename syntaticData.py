import numpy as np
import matplotlib.pyplot as plt
import os
#[Number of trials, episode length, feature dimension]
#torque_npy = np.random.uniform(0, 1, [10, 1000, 6])
if not os.path.exists("syntatic_data"):
    os.mkdir("syntatic_data")

checkpoints=5
num_trials=5
episode_length=1000
feature_dim=6
for chk in range(checkpoints):
    file_name="syntatic_C"+str(chk+1)
    for tr in range(num_trials):
        for i in range(feature_dim):
            omega=0.2*np.random.randn()+0.2
            phi=0.2*np.random.randn()+0.2
            time = np.arange(0, 100, 0.1)
            amplitude = np.sin(omega*time+phi)
            amplitude=np.expand_dims(amplitude,1)
            if i==0:
                per_trial_signal=amplitude
            else:
                per_trial_signal=np.concatenate([per_trial_signal,amplitude],axis=-1)

        if tr==0:
            final_signal=np.expand_dims(per_trial_signal,axis=0)
        else:
            per_trial_signal=np.expand_dims(per_trial_signal,axis=0)
            final_signal = np.concatenate([final_signal, per_trial_signal], axis=0)

    np.save(os.path.join("syntatic_data",file_name+".npy"),final_signal)

gg, ax = plt.subplots(feature_dim,1)
for i in range(feature_dim):
    ax[i].plot(final_signal[0,:,i])
gg.savefig("example_signals.jpg")
plt.show()


