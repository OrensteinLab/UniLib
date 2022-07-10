# Desert Sequence Generator

**Introduction** 

Here I developed a MATLAB code that generated a sequence devoid of any known yeast motifs (downloaded from: YeTFaSCo Version: 1.02), we termed it the desert sequence.

This sequence is the chassis for the synthetic upstream regulatory sequence OL, as well as the LexA bacterial regulatory sequence OL.

Sequence length can be adjustable, the code can be applied on number of shorter sequences and then on the combined sequences, to make a longer sequence (which 

also needs to get checked as desert), I found this makes overall shorter runtimes.


**How it works?**

To generate a desert sequence (devoid from any known yeast motifs, listed in YeastMotifs.xlsx file), start by converting 

the motifs into "regular expression" format, by running the DesertURSGenerator.m code in MATLAB.

This code will generate a regexps_YeastMotifs.xlsx output with all the motifs, in a regular expression description for each motif.

Then, the code randomizes a sequence in any desired length (default: 186bp) and replaces the yeast motifs found in the randomized sequence into a "desert".

The primary code uses the functions: CheckSeqValidity.m, CheckSequence.m, ManualRandSeq.m, and ReplaceMatchedSeq.m automatically.

It is important to save all files indluding functions and scripts in the same path.
 
Once finished, the desert sequence will appear on the MATLAB's command window but also saved in a "Results" .txt file.

