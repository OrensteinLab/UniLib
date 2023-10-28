#general imports
import numpy as np
import pandas as pd
#from deg_project.general import general_utilies

#imports for run_modisco
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from pathlib import Path
import h5py


#imports for genrating modisco dataset
import tensorflow as tf
from tqdm import tqdm
from NN_IG_imp import get_integrated_gradients
import modisco_utilies


#define TF-MoDISco resutls path
modisco_path = 'modisco_files/' #change output folder if needed

#Save PWM patteren
def savePattern(patten,filename,LEN=70):
    raw_data = {'Pos':np.arange(len(patten))+1,'A': patten[:,0],'C': patten[:,1],'G': patten[:,2],'T': patten[:,3]}
    df = pd.DataFrame(raw_data, columns = ['Pos','A','C','G','T'])
    np.savetxt(filename, df.values, fmt='%i\t%0.6f\t%0.6f\t%0.6f\t%0.6f', delimiter="\t", header="Pos\tA\tC\tG\tT",comments='')


def run_modisco(hyp_impscores, impscores, sequences_fetures, null_distribution=None):
    #import TF-MoDISco only here since it's distroying the tf 2 behavior
    import modisco
    import modisco.visualization
    from modisco.visualization import viz_sequence

    #arrange null_distribution as input to the TF-MoDISco if null_distribution exists
    if(null_distribution is None):
        nulldist_args = {}
    else:
        null_distribution = [np.sum(null_distribution_element, axis=1) for null_distribution_element in null_distribution]
        nulldist_perposimp = np.array(null_distribution)
        nulldist_args = {'null_per_pos_scores' : {'task0': nulldist_perposimp}}
    
    #Run TF-MoDISco
    tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
                        #Slight modifications from the default settings
                        target_seqlet_fdr=0.25,
                        seqlets_to_patterns_factory=
                            modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
                                kmer_len=6, num_gaps=1,
                                num_mismatches=0,
                                final_min_cluster_size=60
                            )
                        )(
                    #There is only one task, so we just call this 'task0'
                    task_names=["task0"],
                    contrib_scores={'task0': impscores},                
                    hypothetical_contribs={'task0': hyp_impscores},
                    one_hot=sequences_fetures,
                    **nulldist_args)

    #create Results folder if not exists 
    Path(modisco_path).mkdir(parents=True, exist_ok=True)

    #save the Results
    if(os.path.isfile(modisco_path+"results.hdf5")):
        os.remove(modisco_path+"results.hdf5")

    grp = h5py.File(modisco_path+"results.hdf5")
    tfmodisco_results.save_hdf5(grp)
    hdf5_results = h5py.File(modisco_path+"results.hdf5","r")

    print("Metaclusters heatmap")
    activity_patterns = np.array(hdf5_results['metaclustering_results']['attribute_vectors'])[
                        np.array(
            [x[0] for x in sorted(
                    enumerate(hdf5_results['metaclustering_results']['metacluster_indices']),
                   key=lambda x: x[1])])]
    sns.heatmap(activity_patterns, center=0)
    plt.savefig(modisco_path+"Metaclusters_heatmap.png")

    metacluster_names = [
        x.decode("utf-8") for x in 
        list(hdf5_results["metaclustering_results"]
             ["all_metacluster_names"][:])]

    all_patterns = []
    print(metacluster_names)
    for metacluster_name in metacluster_names:
        print(metacluster_name)
        metacluster_grp = (hdf5_results["metacluster_idx_to_submetacluster_results"]
                                       [metacluster_name])
        print("activity pattern:",metacluster_grp["activity_pattern"][:])
        all_pattern_names = [x.decode("utf-8") for x in 
                             list(metacluster_grp["seqlets_to_patterns_result"]
                                                 ["patterns"]["all_pattern_names"][:])]
        if (len(all_pattern_names)==0):
            print("No motifs found for this activity pattern")
        for i,pattern_name in enumerate(all_pattern_names):
            print(metacluster_name, pattern_name)
            all_patterns.append((metacluster_name, pattern_name))
            pattern = metacluster_grp["seqlets_to_patterns_result"]["patterns"][pattern_name]
            print("total seqlets:",len(pattern["seqlets_and_alnmts"]["seqlets"]))
            background = np.array([0.27, 0.23, 0.23, 0.27])
            print("Hypothetical scores:")
            viz_sequence.plot_weights(pattern["task0_hypothetical_contribs"]["fwd"])
            plt.savefig(modisco_path+'modisco_out/'+'Hypothetical'+str(i)+'.png')
            print("Actual importance scores:")
            viz_sequence.plot_weights(pattern["task0_contrib_scores"]["fwd"])
            plt.savefig(modisco_path+'modisco_out/'+'importance'+str(i)+'.png')
            print("onehot, fwd and rev:")
            viz_sequence.plot_weights(viz_sequence.ic_scale(np.array(pattern["sequence"]["fwd"]),
                                                            background=background)) 
            plt.show()
            plt.savefig(modisco_path+'modisco_out/'+'onehot_fwd'+str(i)+'.png')
            viz_sequence.plot_weights(viz_sequence.ic_scale(np.array(pattern["sequence"]["rev"]),
                                                            background=background))
            plt.savefig(modisco_path+'modisco_out/'+'onehot_rev'+str(i)+'.png')
            savePattern(np.array(pattern["task0_hypothetical_contribs"]["fwd"]),modisco_path+"modisco_out/hyp_pattern"+str(i)+".txt")
            savePattern(np.array(pattern["task0_contrib_scores"]["fwd"]),modisco_path+"modisco_out/imp_pattern"+str(i)+".txt")
            savePattern(np.array(pattern["sequence"]["fwd"]),modisco_path+"modisco_out/onehot_pattern"+str(i)+".txt")

    hdf5_results.close()


#######################################################################################
#compute both importance score and hypothetical importance score
def compute_impratnce_scores(model, sequences_fetures, target_range, batch_size=128):
    ex_list = []
    hyp_ex_list = []
    part_nums = (len(sequences_fetures)//batch_size)+1 #devide the calculation acordding to the bach size 
    for sequences_fetures_part in tqdm(np.array_split(sequences_fetures, part_nums)):
        ex_to_add, hyp_ex_to_add = get_integrated_gradients(model, sample_inputs=sequences_fetures_part, target_range=target_range, multiple_samples=True)
        ex_list += list(ex_to_add)
        hyp_ex_list += list(hyp_ex_to_add)
    return ex_list, hyp_ex_list

#create null distribution for modisco
#DAN: ignore null distribution for now
def create_null_distribution (model, sequences_fetures, intial_features, target_range, batch_size): 
    if (create_null_distribution):
        null_distribution = []
        sequences_fetures_for_dinuc_shuffle, _,  intial_features_dinuc_shuffle, _ = train_test_split(sequences_fetures, intial_features, test_size=0.5, random_state=42)
        for i in range(len(sequences_fetures_for_dinuc_shuffle)):
            sequences_fetures_for_dinuc_shuffle [i] = modisco_utilies.dinuc_shuffle(sequences_fetures_for_dinuc_shuffle[i], seed=42)
        
        part_nums = (len(sequences_fetures_for_dinuc_shuffle)//batch_size)+1
        for sequences_fetures_for_dinuc_shuffle_part, intial_features_dinuc_shuffle_part in tqdm(zip(np.array_split(sequences_fetures_for_dinuc_shuffle, part_nums), np.array_split(intial_features_dinuc_shuffle, part_nums))):
            for i  in range(len(model)):
                ex_to_add, _hyp_ex_to_add = get_integrated_gradients(model[i], sample_inputs=[sequences_fetures_for_dinuc_shuffle_part, intial_features_dinuc_shuffle_part], target_range=target_range, multiple_samples=True)
                ex_to_add = ex_to_add[0] #[0] for taking only sequence importance
                ex = ex_to_add if i == 0 else ex + ex_to_add

            ex = ex/len(model) #take the mean
            null_distribution = null_distribution + list(ex)
    else:
        null_distribution = None
    
    return null_distribution

#ex_list, hyp_ex_list = compute_impratnce_scores(model, sequences_fetures, target_range, batch_size=128)
#run_modisco(hyp_ex_list, ex_list, sequences_fetures)
