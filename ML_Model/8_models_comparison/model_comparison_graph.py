import numpy as np
import matplotlib.pyplot as plt

min_readtot= [8769, 5842, 4312, 3296, 2559, 1987, 1531, 1162, 854, 601, 391, 216, 71, 1]
amount_of_data_points= [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]

x_labels = [f"{amount}\n({min_read}) " for amount, min_read in zip(amount_of_data_points, min_readtot)]

#pearson corr
pc_bins_116_weights = [0.35811884055486004, 0.3766243706411868, 0.3635014757887214, 0.37775563585417826, 0.36965376250138293, 0.3785254544579794, 0.3581144082894784, 0.34548954102681606, 0.3285801781650062, 0.3010052603842266, 0.2675738445780359, 0.23359874840596578, 0.1976891075475296, 0.18626241815436878]
pc_bins_116 =[0.344740475380109, 0.39008435222693716, 0.38353016385174643, 0.33856460855582704, 0.35772580178453095, 0.3782581996833886, 0.3550623110499892, 0.3457225517021286, 0.33063824343596626, 0.2896961359059638, 0.2525858112398219, 0.22224937333619715, 0.15001736064895713, 0.14443607837332004]
pc_bins_101_weights= [0.34697898671562355, 0.34678441394070114, 0.3733227646816462, 0.3778449024159993, 0.3791607409995943, 0.37486352609300055, 0.3631390706368026, 0.350781488211409, 0.3302662370479549, 0.28996796931520685, 0.2807854186277306, 0.23434163171333022, 0.21018891809176, 0.1935677782528874]
pc_bins_101= [0.30716465115210134, 0.38460482736144763, 0.373673609056842, 0.3799931677308087, 0.36736001796121653, 0.37330434575869886, 0.3630881399399228, 0.35231872400596803, 0.3212657383467425, 0.2923724124960909, 0.25158862744037325, 0.21100940751597963, 0.15928321597368666, 0.16662779598982258]
pc_meanfl_116 = [0.3790811432782878, 0.4058749356226373, 0.4185169402751928, 0.42137245535156764, 0.4108169521474553, 0.39971626339395355, 0.37516563577516004, 0.36463894013398535, 0.3380963992819762, 0.3025622475834313, 0.2645181488353845, 0.21101226558586253, 0.16423077216964171, 0.14034052364506155]
pc_meanfl_116_weights = [0.38752297938480335, 0.4415025430729681, 0.4186679490634201, 0.41854530169681314, 0.4112007848062857, 0.3902486248408761, 0.3897504009524424, 0.3600860605535796, 0.34313902860539525, 0.3100008398020513, 0.26234064653761296, 0.23428969798761679, 0.19133329252252007, 0.18969091578113817]
pc_meanfl_101_weights= [0.38984244536445045, 0.3919640893177209, 0.40539164450233905, 0.4035384793864891, 0.4004164316905963, 0.3919803116508077, 0.3639146917951491, 0.36608313486476796, 0.3230380650541081, 0.2840643066851932, 0.27788972621959673, 0.25138330722996083, 0.2137967837326809, 0.19162202093354186]
pc_meanfl_101= [0.3860313675265291, 0.4105006230788281, 0.37840519187072513, 0.4025885292681197, 0.3939019183517708, 0.3881958828401708, 0.37769268991265925, 0.35249844752121895, 0.332066389737541, 0.313132643535983, 0.25782520570897594, 0.19653202279455198, 0.15427794873249467, 0.16956058181998926]

fig= plt.figure(figsize=(16, 12))
fig.subplots_adjust(top=0.95, left=0.1, right=0.9, bottom=0.18)

# Plot the lines with different colors
plt.plot(amount_of_data_points, pc_meanfl_116_weights, 'g.-', label="Mean FL,116bp sequence, with sample weights model fit")
plt.plot(amount_of_data_points, pc_meanfl_116, 'b.-', label="Mean FL,116bp sequence")
plt.plot(amount_of_data_points, pc_meanfl_101_weights, 'r.-', label="Mean FL,101bp sequence, with sample weights model fit")
plt.plot(amount_of_data_points, pc_meanfl_101, 'c.-', label="Mean FL,101bp sequence")
plt.plot(amount_of_data_points, pc_bins_116_weights, 'y.-', label="4 bins,116bp sequence, with sample weights model fit")
plt.plot(amount_of_data_points, pc_bins_116, 'm.-', label="4 bins,116bp sequence")
plt.plot(amount_of_data_points, pc_bins_101_weights, 'k.-', label="4 bins,101bp sequence, with sample weights model fit")
plt.plot(amount_of_data_points, pc_bins_101, 'orange', label="4 bins,101bp sequence")

plt.xticks(amount_of_data_points, x_labels, fontsize=16, rotation=40, ha='right')
plt.yticks(np.arange(0, 0.5, step=0.05), fontsize=16)
plt.xlabel("# Variant in the training set \n (Total read cutoff)", fontsize=25)
plt.ylabel("Pearson correlation", fontsize=25)

# Set legend with custom placement and font size
legend = plt.legend(loc='lower left', fontsize=18)
legend.set_title("Legend", prop={"size": 18})

plt.savefig('models_comp_graph.png', dpi=300)
plt.show()



