%This code performs 2 main things:
%1. Converts the DNA binding sites/motifs to a "regular expression" (using
%regexp) format.
%2. Randomizes a sequence in a desired length and eliminates the motifs from the forst part of the code
%until no motifs are found in the sequence, and lastly- the resultant sequence can be termed "Desert".
clc
clear

%First part- converting yeast motifs:
% Extracting the yeast motifs:
sheet = 'allData';
xlRange = 'D2:D3775';
[ ndata, text, alldata] = xlsread('YeastMotifs.xlsx', sheet, xlRange); %I will use only the 'text' array
filename = 'regexps_YeastMotifs.xlsx';
offset=0;
regexps={};
shorts=[]; % indices of too-short motifs
lowers=[]; % indices of lower-case motifs
wb = waitbar(0,'Please Wait...');
textLength = size(alldata,1);
for ii=1:textLength
    tmp=alldata{ii};
    if(isnan(tmp)) %shorter motifs (less than 6 nucleotides) are overlooked.
        shorts=[shorts ii];
        continue;
    end
    if(sum(isstrprop(tmp, 'lower'))==size(tmp,2))%lower-case motifs are overlooked.
        lowers=[lowers ii];
        continue;
    end
    rr=''; % empty regexp
    % build regexp
    for jj=1:size(tmp,2)
        switch(tmp(jj))
            case 'A'
                rr=[rr 'A'];
            case 'T'
                rr=[rr 'T'];
            case 'G'
                rr=[rr 'G'];
            case 'C'
                rr=[rr 'C'];
            case 'W'
                rr=[rr '[AT]{1,1}'];
            case 'S'
                rr=[rr '[GC]{1,1}'];
            case 'R'
                rr=[rr '[GA]{1,1}'];
            case 'Y'
                rr=[rr '[TC]{1,1}'];
            case 'M'
                rr=[rr '[AC]{1,1}'];
            case 'K'
                rr=[rr '[GT]{1,1}'];
            case 'B'
                rr=[rr '[GTC]{1,1}'];
            case 'D'
                rr=[rr '[GTA]{1,1}'];
            case 'H'
                rr=[rr '[CTA]{1,1}'];
            case 'V'
                rr=[rr '[CGA]{1,1}'];
            case {'N','x','a','c','g','t'}
                rr=[rr '[TCGA]{1,1}'];
            otherwise
                disp(['string ans' num2str(ii) ' pos ' num2str(jj) ' PROBLEM!!!!!\n']);
                break;
        end
    end
    regexps{ii}=rr;
    offset=offset+1;
    xlswrite(filename,regexps{ii},1, sprintf('A%d',offset))
    waitbar(ii/textLength,wb)
end
close(wb)
fprintf(1, 'Finished regexp part!!!!!!!!!\n')

%Second part- generating a desert sequence:
found=0;
FIN_NUM=size(regexps,2);
loopnum=0;
loopMaxNum = 1000000; %maximal number of random sequences
counter=0;
t = datetime('now');
fid= fopen(['Results ',strrep(datestr(t),':','_'),'.txt'],'w');
wb = waitbar(0,'Please Wait...');

seqLength =186; %Here the length of the sequence can be changed.
desertUAS = ManualRandSeq(seqLength)  % Generates 1 random sequence 

% ------ Load chirp sound -----------
chirpData = load('chirp.mat');
chirpObj = audioplayer(chirpData.y,chirpData.Fs);

%check random sequence for yeast motifs and change every yeast motif to a desert. 
while found~=1
    CheckSequence; 
end
close(wb)
fclose(fid);
fprintf('Minimum matches:%d\n',minSeq)
fprintf('Optimal sequence: %s\n',minSeqStr) 
fprintf(1, 'Finished script- found a desert sequence at last!!!!!!!!!!!!\n')

