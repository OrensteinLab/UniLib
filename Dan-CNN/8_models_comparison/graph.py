import numpy as np
import matplotlib.pyplot as plt


amount_of_data_points= [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
min_readtot = [8957, 6009, 4447, 3418, 2671, 2084, 1623, 1245, 934, 670, 454, 271, 119, 4]


#4 bins
bins_101 = [0.43921874232785374, 0.44286284421504263, None, None, 0.40805303410145455, 0.4028971010679241, 0.3928478565275008, 0.3667106560825276, 0.37656429269766284, 0.3346718376580219, 0.31938213325102105, 0.24621832228379834, 0.21387377643257244, 0.22015506020283265]
bins_116 = [0.4439650800010919, 0.42151971724646586, 0.4270775971861523, None, None, 0.4054778192463313, 0.3821858148804287, 0.37474176566201445, 0.36925514637678336, 0.34208641415609675, 0.28990041903329605, 0.24643574979200106, 0.18329922346339117, 0.16140957280868912]
bins_101_weights = [0.44114952811908825, 0.4463873054110701, 0.43636829096468244, None, None, 0.3999228411313246, 0.3959662816325887, 0.3745770781028663, 0.3666363874027931, 0.34452796302001754, 0.32229018732723175, 0.27135947439494834, 0.215393583806063, 0.2942022716583792]
bins_116_weights = [0.42813257612426037, 0.4397767222770199, 0.435341918421407, None, 0.4227549512323934, 0.4103730333494126, 0.382280451050999, 0.3900729134358011, 0.36408796462362214, 0.3327079899596418, 0.32322293326327267, 0.2737769977104497, 0.26264802883053834, 0.22870101160384462]

#meanFL
meanfl_116 = [0.42545377889645447, 0.4517120992311476, 0.44790163709721587, 0.4188575828912293, 0.43576296680173393, 0.4214585842520622, 0.4008469785153387, 0.3731506290772244, 0.3498895013276486, 0.3054362210299629, 0.279929925859923, 0.2002524397410897, 0.16280524523649853, 0.14385669607565915]# amount_of_data_points = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
meanfl_101 = [0.4430852154160887, 0.4509694804524299, 0.45093567791092903, 0.4433060538432929, 0.41084638771737847, 0.4208453726868934, 0.39647527102645336, 0.3689539757657422, 0.3412701251899021, 0.326218807640125, 0.2950895416412297, 0.2157644049262403, 0.15034433122292876, 0.15108644222751635]# amount_of_data_points = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
meanfl_116_weights = [0.4267723077715702, 0.44720297379223367, 0.4386681616618783, 0.4279744910707042, 0.43179092007028474, 0.4167435899354919, 0.394965480198018, 0.37082469766110926, 0.3355747956773678, 0.328994375831432, 0.27654980986488475, 0.2370162677820199, 0.20132151670936904, 0.11241139872346481]# amount_of_data_points = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
meanfl_101_weights = [0.4511221716051027, 0.4388472358804014, 0.43215852308305347, 0.42781734057500515, 0.43558375328005094, 0.4103491983428952, 0.3875244178283658, 0.36764986074477646, 0.3479813950095468, 0.31016279755172954, 0.27027497106355813, 0.2287042342539751, 0.18396920873335057, 0.15391673581233706]# amount_of_data_points = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]



# print(len(pearson_corr), len(amount_of_data_points), len(k_fold_pearson_std))
#
# for item in min_readtot:
#     print(item, end=")      (")
fig= plt.figure(figsize=(16, 12), dpi=300)

plt.plot(amount_of_data_points, bins_116, 'b*--',label="4 bins,116 seq (with barcode)")
plt.plot(amount_of_data_points, meanfl_116, 'g*-', label="meanFL, 116 seq (with barcode)")
plt.plot(amount_of_data_points, bins_101, 'k*-', label="4 bins,101 seq (without barcode)")
plt.plot(amount_of_data_points, meanfl_101, 'b.-', label="meanFL, 101 seq (without barcode)")
plt.plot(amount_of_data_points, bins_116_weights, 'c.-',label="4 bins,116 seq (with barcode), with sample weights model fit")
plt.plot(amount_of_data_points, meanfl_116_weights, 'y.-', label="meanFL, 116 seq (with barcode), with sample weights model fit")
plt.plot(amount_of_data_points, bins_101_weights, 'y.--', label="4 bins,101 seq (without barcode), with sample weights model fit")
plt.plot(amount_of_data_points, meanfl_101_weights, 'r.-', label="meanFL, 101 seq (without barcode), with sample weights model fit")

plt.xticks(amount_of_data_points)
plt.ylabel("pearson correlation", fontsize=14)
plt.title("models comparison: 4 bins/meanFL, with/without barcode, with/without sample_weights model fit")

# plt.xlabel("\n\namount of 116 sequences data points (minimum readtot of the data points)", fontsize=14)
plt.legend()
plt.show()
plt.savefig('models_comp.png',dpi=300)