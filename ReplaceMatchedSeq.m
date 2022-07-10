function newSeq = ReplaceMatchedSeq(desertUAS, matchedSeq)
for i=1:length(matchedSeq)
    singleMatchedSeq = matchedSeq{i};
    if (iscell(singleMatchedSeq))
        for j=1:length(singleMatchedSeq)
            innerMatchedSeq = singleMatchedSeq{j};
            desertUAS = regexprep(desertUAS, innerMatchedSeq, ManualRandSeq(length(innerMatchedSeq)));
%             ind = strfind(desertUAS, innerMatchedSeq);
%             wordLength = length(innerMatchedSeq);
%             desertUAS(ind:ind+wordLength-1) = ManualRandSeq(wordLength);
        end
        continue
    end
    singleMatchedSeq = singleMatchedSeq{1};
    desertUAS = regexprep(desertUAS, singleMatchedSeq, ManualRandSeq(length(singleMatchedSeq)));
%     ind = strfind(desertUAS, singleMatchedSeq);
%     wordLength = length(innerMatchedSeq);
%     desertUAS(ind:ind+wordLength-1) = ManualRandSeq(wordLength);
end
newSeq = desertUAS;
end