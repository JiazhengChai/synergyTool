import os
import argparse
from matplotlib import pyplot as plt
from utils import R2,plot_spatial_W,plot_spatiotemporal_W
from core import calculate_synergy
import numpy as np

##README##
# 1) Run this file.
# 2) Read the different arguments that are available for parsing.
# Note: parameters to play with: windows, start_index, synergy_type

parser = argparse.ArgumentParser()

parser.add_argument('--folder_path',
                    type=str,
                    default="syntatic_data",
                    help='The path to the folder containing all the time series numpy files.'
                         'To use this file, the name of each file must contain a C+number to'
                         'identify the checkpoint ', )
parser.add_argument('--output_folder_name',
                    type=str,
                    default='output_figure',
                    help='the name of the folder for output figures')
parser.add_argument('--windows',
                    type=int,
                    default=300,
                    help='the window size to truncate the input time series for analysis')
parser.add_argument('--start_index',
                    type=int,
                    default=0,
                    help='The time index from which the signal will be retained for analysis.'
                         'Signals before the start_index will be truncated.')
parser.add_argument('--synergy_type',
                    type=str,choices=['spatial','spatiotemporal','temporal'],
                    default='spatiotemporal',
                    help='the synergy type')
parser.add_argument('--no_label',
                    action='store_false',
                    help='boolean to state if label of curves is desired in plots')
parser.add_argument('--y_lim',
                    type=float,
                    default=None,#9.1
                    help='the limit of the Y-axis for the surface area plot'
                    )

cmap = plt.cm.viridis
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplen = len(cmaplist)

plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 25
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.family'] = 'Times New Roman'
LW = 3

args = parser.parse_args()
windows=args.windows
start_index=args.start_index
synergy_type=args.synergy_type
output_folder_name=args.output_folder_name
with_label=args.no_label
y_lim=args.y_lim

folder_path=args.folder_path

##############files ordering#############
all_npy=[]
for name in os.listdir(folder_path):
    if name[-3::] == 'npy':
        name_elements=name.split('_')
        checkpoint_identifier=None
        for elem in name_elements:
            if "C" in elem:
                if(".npy") in elem:
                    elem=elem.replace('.npy',"")
                checkpoint_identifier=elem
        if not checkpoint_identifier:
            raise("File name must contain C+number to be able to identify files of different chekpoints.")

        all_npy.append(int(checkpoint_identifier.replace('C', '')))

new_index=sorted(range(len(all_npy)), key=lambda k: all_npy[k])
ori_files=(os.listdir(folder_path))
sorted_files=[]
for ni in new_index:
    sorted_files.append(ori_files[ni])
########################################

total_checkpoints=len(all_npy)
agent_name = folder_path.split("/")[-1]

step = cmaplen // total_checkpoints
color_list = []
c = cmaplen - 1
for l in range(total_checkpoints):
    color_list.append(cmaplist[c])
    c -= step

all_R2_list = []
all_surface_area=[]
r_sq_all_combare, r_sq_all_combare_ax = plt.subplots(1, 1)
SA_plot, SA_plot_ax = plt.subplots(1, 1)
all_label=[]

for count,npy_file in enumerate(sorted_files):

    npy_file_path=os.path.join(folder_path,npy_file)
    torque_npy = np.load(npy_file_path)
    #print(torque_npy.shape)

    ####call the funciton from core.py#############
    R2_list_single_line, surface_area_single_line,W_list=calculate_synergy(torque_npy,synergy_type, start_index,windows)
    ###############################################

    all_label.append('Level ' + str((count+1)))

    all_R2_list.append(R2_list_single_line)
    all_surface_area.append(surface_area_single_line)

    if synergy_type=="spatial":
        for ind, w_mat in enumerate(W_list):
            plot_spatial_W(w_mat,ind + 1,npy_file.split(".")[0],output_folder_name,format=".jpg",save=True,scale=[-1,1])
    elif synergy_type=="spatiotemporal":
        for ind, w_mat in enumerate(W_list):
            plot_spatiotemporal_W(w_mat,torque_npy.shape[-1],ind + 1,npy_file.split(".")[0],output_folder_name,format=".jpg",save=True)


path = output_folder_name
os.makedirs(path, exist_ok=True)

for index,ral in enumerate(all_R2_list):
    r_sq_all_combare_ax.plot(range(1, len(ral) + 1), ral, color=color_list[index],
                             label=all_label[index])

    if index == 0:
        r_sq_all_combare_ax.set_ylabel(r"${0:s}$".format(R2()))
        r_sq_all_combare_ax.set_xlabel('Number of principal components')
        r_sq_all_combare_ax.set_title(synergy_type+ ' R2 vs Number of principal components')

SA_plot_ax.plot(range(1,total_checkpoints+1),all_surface_area)
SA_plot_ax.set_ylabel("Synergy level")
SA_plot_ax.set_xlabel('Checkpoints')
SA_plot_ax.set_title(synergy_type+ ' surface area under curves')

SA_plot.tight_layout()
if y_lim:
    SA_plot_ax.set_ylim([0,y_lim])

r_sq_all_combare.tight_layout()
r_sq_all_combare_ax.set_ylim([0, 1.05])
if not with_label:
    r_sq_all_combare.savefig(os.path.join(path,  agent_name+'_'+synergy_type+'_synergy_windows_'+str(windows)+'_start_'+str(start_index)+'.png'))

else:
    r_sq_all_combare_ax.legend(loc=4,prop={'size': 15})
    r_sq_all_combare.savefig(os.path.join(path, agent_name+'_'+synergy_type+'_synergy_windows_'+str(windows)+'_start_'+str(start_index)+'_labeled.png'))

SA_plot.savefig(os.path.join(path, agent_name+'_'+synergy_type+'_synergy_level_windows_'+str(windows)+'_start_'+str(start_index)+'.png'))
#plt.show()
