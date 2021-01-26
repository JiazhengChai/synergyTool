import numpy as np
from sklearn.decomposition import TruncatedSVD,PCA
from scipy import integrate
from sklearn.metrics import r2_score
from utils import reshape_into_spt_shape,reshape_into_spatial_shape,reshape_into_temporal_shape
import matplotlib.pyplot as plt
##README##
# 1) Run this file.
# 2) Read the description inside the calculate_synergy function.

def calculate_synergy(torque_npy,synergy_type="spatiotemporal", start_index=0,windows=300):
    """
    This is the core function to calculate synergy data from the time series provided.
    SVD is used instead of PCA because of stability issue.
    Please run this file to get an idea of the usage of this function.

    :param torque_npy: The time series to extract synergy form.
                        Shape=[Number of trials, episode length, feature dimension]
    :param synergy_type: The type of synergy. Choice: ["spatial","spatiotemporal","temporal"]
    :param start_index: The time index from which the signal will be retained for analysis.
                        Signals before the start_index will be truncated.
                        This is mainly to discard signals in the beginning transient state.
    :param windows:  The windows size of signal truncation for synergy extraction.
    :return: R2_list_single_line    :A list of R2 reconstruction accuracy for the time series when the PCA
                                     components vary from 1 to n for the reconstruction.
            surface_area_single_line:The surface area under the R2 reconstruction curve.
    """

    total_trials = torque_npy.shape[0]
    max_episode_length = torque_npy.shape[1]
    total_action_dim = torque_npy.shape[2]

    if (start_index + windows) > max_episode_length:
        print("Warning : start_index + windows is longer than the maximum rollout length.")
        print("Warning : Please adjust the start_index or windows.")
        exit()
    else:
        X=torque_npy[:,start_index:start_index+windows,:]

    rsq_single_list=[]
    W_list=[]
    if synergy_type=='spatiotemporal':
        mx = np.mean(X, axis=1)

        for k in range(X.shape[1]):
            X[:, k, :] = X[:, k, :] - mx

        X = reshape_into_spt_shape(X)#4,2400

        num_features = X.shape[1]
        num_vec_to_keep_max = X.shape[0] + 1
        extract_method = TruncatedSVD

    elif synergy_type=='spatial':

        X = reshape_into_spatial_shape(X)

        mx = np.mean(X, axis=1)

        X = X - np.expand_dims(mx, 1)
        X=X.T #1200,8

        num_features = X.shape[1]
        num_vec_to_keep_max=X.shape[1]+1
        extract_method=PCA

    elif synergy_type == 'temporal':
        mx = np.mean(X, axis=1)

        for k in range(X.shape[1]):
            X[:, k, :] = X[:, k, :] - mx

        X = reshape_into_temporal_shape(X)
        X=X.T#32,300

        num_features = X.shape[1]
        num_vec_to_keep_max = X.shape[0] + 1
        extract_method = TruncatedSVD


    for num_vec_to_keep_ in range(1,num_vec_to_keep_max):
        try:
            pca = extract_method(n_components=num_vec_to_keep_)
            pca.fit(X)
        except:
            pca = extract_method(n_components=num_vec_to_keep_)
            pca.fit(X)

        eig_vecs = pca.components_
        eig_vals = pca.singular_values_
        eig_pairs = [(eig_vals[i], eig_vecs[i, :]) for i in range(len(eig_vals))]
        percentage = sum(pca.explained_variance_ratio_)
        proj_mat = eig_pairs[0][1].reshape(num_features, 1)

        for eig_vec_idx in range(1, num_vec_to_keep_):
            proj_mat = np.hstack((proj_mat, eig_pairs[eig_vec_idx][1].reshape(num_features, 1)))

        W = proj_mat
        W_list.append(W)

        C = X.dot(W)
        X_prime = C.dot(W.T)

        if synergy_type=="spatiotemporal":
            Vm = np.mean(X, axis=0, keepdims=True)
            resid = X - np.dot(Vm, np.ones((X.shape[1], 1)))

        elif synergy_type=="spatial":
            Vm = np.mean(X, axis=0, keepdims=True)
            resid = X - np.dot( np.ones((X.shape[0], 1)),Vm)

        elif synergy_type == "temporal":
            Vm = np.mean(X, axis=0, keepdims=True)
            resid = X - np.dot(np.ones((X.shape[0], 1)), Vm)

        resid2 = X - X_prime
        SST = np.linalg.norm(resid) ** 2
        SSE = np.linalg.norm(resid2) ** 2
        rsq = 1 - SSE / SST

        rsq_single_list.append(rsq)

    R2_list_single_line=rsq_single_list
    surface_area_single_line=integrate.simps(rsq_single_list,range(1,num_vec_to_keep_max))

    return R2_list_single_line,surface_area_single_line,W_list

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils import R2

    all_surface_area=[]
    all_R2_line=[]
    for i in range(10):
        torque_npy = np.random.uniform(0, 1, [10, 1000, 6])
        if i==0:
            torque_plot, torque_plot_ax = plt.subplots(6, 1)

            for i in range(torque_npy.shape[-1]):
                torque_plot_ax[i].set_ylim([0, 1])
                torque_plot_ax[i].plot(torque_npy[0,900::,i])
                torque_plot_ax[i].set_ylabel("Joint "+str(i))

                if i == 0:
                    torque_plot_ax[i].set_title("Random series")

                if i == torque_npy.shape[-1]-1:
                    torque_plot_ax[i].set_xlabel('time')

        R2_list_single_line, surface_area_single_line,W_list = calculate_synergy(torque_npy,synergy_type="spatiotemporal")#"spatial""temporal""spatiotemporal"
        all_surface_area.append(surface_area_single_line)
        all_R2_line.append(R2_list_single_line)

    r_sq_all_combare, r_sq_all_combare_ax = plt.subplots(1, 1)
    for ind,line in enumerate(all_R2_line):
        r_sq_all_combare_ax.plot(range(1, len(line) + 1), line)
        if ind==0:
            r_sq_all_combare_ax.set_ylabel(r"${0:s}$".format(R2()))
            r_sq_all_combare_ax.set_xlabel('Number of principal components')
            r_sq_all_combare_ax.set_title("R2 of random series")

    SA_plot, SA_plot_ax = plt.subplots(1, 1)
    SA_plot_ax.plot(range(1, len(all_surface_area) + 1), all_surface_area)
    SA_plot_ax.set_ylabel("Synergy level")
    SA_plot_ax.set_xlabel('Checkpoints')
    SA_plot_ax.set_title('Surface area under R2 lines')

    SA_plot.tight_layout()

    plt.show()