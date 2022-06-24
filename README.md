# UniLib
**intro**

We present a CNN desinged to predict the fluorescence levels on a scale of 1-4 of a RNA sequences.
As a part of this study we generated dataset comprising of around 150k RNA sequences, each sequence is 116 long with the first 15 nt being a barcode,
it is possible to ignore the first 15 nt (we discovered it doesnt change the results).

This repository contains 4 different models:

  * bins - this model's output is a 1x4 probabilty vector 
  
  * bins_sample_weights - similar to "bins" model but is takes into consideration the amount of total reads each sequence has

  * meanFL - the output of this model is a single scalar value called meanFL calculated by the formula : p(bin1)*607 + p(bin2)*1364 + p(bin3)*2596 + p(bin4)*7541       (p is the probabilty of the sequence being in the x bin)

  * meanFL_sample_weights - similar to "meanFL" model but is takes into consideration the amount of total reads each sequence has 

**Getting started**

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

**Prerequisites**

  * pandas
  
  * numpy
  
  * tensorflow
  
  * keras
  
  * scipy
  
  * sklearn
  
  **How it works**
  
  Training the model:
      
