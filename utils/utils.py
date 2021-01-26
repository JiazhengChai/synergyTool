import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def R2():

    return r'R^{{{e:d}}}'.format(e=int(2))

def my_as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'\times 10^{{{e:d}}}'.format(e=int(e))

def reshape_into_spt_shape(X_input):
    assert len(X_input.shape) == 3
    trial_num, sample_l, action_dim = X_input.shape

    X_loc = np.expand_dims(X_input[0, :, 0], 0)

    for i in range(1, action_dim, 1):
        X_loc = np.concatenate((X_loc, np.expand_dims(X_input[0, :, i], 0)), axis=1)

    for l in range(1, trial_num, 1):
        X_inner = np.expand_dims(X_input[l, :, 0], 0)
        for i in range(1, action_dim, 1):
            X_inner = np.concatenate((X_inner, np.expand_dims(X_input[l, :, i], 0)), axis=1)

        X_loc = np.concatenate((X_loc, X_inner), axis=0)

    return X_loc

def reshape_into_temporal_shape(X_input):
    assert len(X_input.shape) == 3
    trial_num, sample_l, action_dim = X_input.shape

    X_loc = np.expand_dims(X_input[0, :, 0], -1)

    for i in range(1, action_dim, 1):
        X_loc = np.concatenate((X_loc, np.expand_dims(X_input[0, :, i], -1)), axis=1)

    for l in range(1, trial_num, 1):
        X_inner = np.expand_dims(X_input[l, :, 0], -1)
        for i in range(1, action_dim, 1):
            X_inner = np.concatenate((X_inner, np.expand_dims(X_input[l, :, i], -1)), axis=1)

        X_loc = np.concatenate((X_loc, X_inner), axis=-1)

    return X_loc

def reshape_into_spatial_shape(X_input):
    assert len(X_input.shape) == 3
    trial_num, sample_l, action_dim = X_input.shape

    X_loc = X_input[0, :, :]
    X_loc=np.transpose(X_loc)

    for l in range(1, trial_num, 1):
        X_loc = np.hstack((X_loc,np.transpose(X_input[l, :, :])))

    return X_loc

def plot_spatial_W(w_matrix,num_vec_to_keep,name,save_folder,format=".jpg",save=False,scale=[-1,1]):
    joint_list = []

    for i in range(w_matrix.shape[0]):
        joint_list.append("Ch "+str(i+1))

    joint_list.append("")
    joint_list=joint_list[::-1]

    gg, ax = plt.subplots(1,num_vec_to_keep,figsize=(4*(num_vec_to_keep-1)+6,8))

    if num_vec_to_keep==1:
        ax.barh(range(w_matrix.shape[0]), w_matrix[::-1, 0], 0.8)

        ax.set_xlabel('$W_1$' )
        ax.get_yaxis().set_visible(True)
        ax.set_yticklabels(joint_list)
        ax.set_xlim(scale)
    else:
        for i in range(num_vec_to_keep):
            ax[i].barh(range(w_matrix[:, i].shape[0]), w_matrix[::-1, i], 0.8)

            ax[i].set_xlabel('$W_' + str(i+1)+'$')
            ax[i].get_yaxis().set_visible(False)

            if i == 0:
                ax[i].get_yaxis().set_visible(True)
                ax[i].set_yticklabels(joint_list)
            ax[i].set_xlim(scale)

    gg.tight_layout()

    if save == False:
        plt.show()
    else:
        path = save_folder+'/spatial_W_matrix'+'/'+'PCA_components_'+name
        os.makedirs(path, exist_ok=True)
        gg.savefig(path+'/PCA_components_'+str(num_vec_to_keep) + format,format=format.split(".")[-1])
        plt.close(gg)

def plot_spatiotemporal_W(w_matrix,total_vec,num_vec_to_keep,name,save_folder,format=".jpg",save=False,scale=[-0.1,0.1]):
    gb, bx = plt.subplots(total_vec, num_vec_to_keep, figsize=(2*(num_vec_to_keep-1)+6, 15))

    sample_length = int(w_matrix.shape[0] / total_vec)
    if num_vec_to_keep == 1:
        for k in range(total_vec):
            bx[k].plot(w_matrix[sample_length * k:sample_length * (k + 1), 0],
                          linewidth=3)  # ,c='r',alpha=0.5
            bx[k].axhline(0, color='black', alpha=0.5, linestyle='--')

            bx[k].get_xaxis().set_visible(False)
            bx[k].get_yaxis().set_visible(False)

            bx[k].set_ylim(scale)

            if k == 0:
                bx[k].set_title('W' + '$_{}$'.format(1))

            if k == total_vec - 1:
                bx[k].get_xaxis().set_visible(True)
                #bx[k].xaxis.set_major_locator(MaxNLocator(integer=True))
                # if j == 1:
                bx[k].set_xlabel('Time steps')

            bx[k].get_yaxis().set_visible(True)
            bx[k].set_ylabel('Ch {}'.format(k + 1))
    else:
        for j in range(num_vec_to_keep):
            for k in range(total_vec):
                bx[k, j].plot(w_matrix[sample_length * k:sample_length * (k + 1), j],
                              linewidth=3)  # ,c='r',alpha=0.5
                bx[k, j].axhline(0, color='black', alpha=0.5, linestyle='--')

                bx[k, j].get_xaxis().set_visible(False)
                bx[k, j].get_yaxis().set_visible(False)

                bx[k, j].set_ylim(scale)

                if k == 0:
                    bx[k, j].set_title('W' + '$_{}$'.format(j + 1))

                if k == total_vec - 1:
                    bx[k, j].get_xaxis().set_visible(True)
                    #bx[k, j].xaxis.set_major_locator(MaxNLocator(integer=True))
                    if j == 1:
                        bx[k, j].set_xlabel('Time steps')
                if j == 0:
                    bx[k, j].get_yaxis().set_visible(True)
                    bx[k, j].set_ylabel('Ch {}'.format(k + 1))


    gb.tight_layout()
    if save == False:
        plt.show()
    else:
        path = save_folder+'/spatiotemporal_W_matrix'+'/'+'PCA_components_'+name
        os.makedirs(path, exist_ok=True)
        gb.savefig(path+'/PCA_components_'+str(num_vec_to_keep) + format,format=format.split(".")[-1])
        plt.close(gb)

