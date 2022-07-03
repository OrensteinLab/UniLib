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
  
  **Training the model:**
  
  * In order to train the model on the same data:
  
  first you need to choose a model from the 8 models we compared then you simply have to replace this line:
  
  ![image](https://user-images.githubusercontent.com/101515707/177045823-2edb66a4-3a76-48df-b430-8dbaf21a93c7.png)
    
    with a line that loads the data to your machine.
    
  * To train the model on another set of data you need 
  
  A) To choose a model and make sure your data contains the relevant features e.g bin readings or meanFL
  
  B) Replace this line:
  
  ![image](https://user-images.githubusercontent.com/101515707/177045823-2edb66a4-3a76-48df-b430-8dbaf21a93c7.png)
  
  with a line that loads your own data.
  
  C) Make sure the labels are loaded correctly into the following veriables:
  
  ![image](https://user-images.githubusercontent.com/101515707/177046169-451b154b-f47a-45c6-b512-eb612a97f395.png)
  
  **Getting predictions:**
  
  A) Transform the sequences into a one hot matrix representation of the string using the oneHotDeg function:
  ![image](https://user-images.githubusercontent.com/101515707/177046458-7cfd1ac1-04b9-4642-8ad3-d47767c9e2a4.png)

  B) Use model.predict(your_matrix_goes_here)
  
  
  
  
  
  

  
