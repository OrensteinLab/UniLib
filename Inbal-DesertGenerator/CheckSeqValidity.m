function [foundSequances ] = CheckSeqValidity(seq, regExpCellArray )
%This function checks whether multiple sub-sequances given by "regExpCellArray" can
%be found in a sequance given by "seq"
out = regexp(seq,regExpCellArray,'match'); %looks for matches between seq and regexpsarray
matchedFound = cellfun(@(x) ~isempty(x),out);
foundSequances = out(matchedFound);
end

