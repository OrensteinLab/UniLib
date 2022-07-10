stuckSequenceCounter = 0; 
minSeq = 9999;
matchedSeq_i = cell(9999,1);
while (stuckSequenceCounter<loopMaxNum)
    loopnum=loopnum+1;
    tic; 
    matchedSeq_ii = CheckSeqValidity(desertUAS, regexps);
    % ------------- Checks whether the sequence is a "stuck" sequence ------
    %--------------- and that the number of matched sequences reduces ------
    if (length(matchedSeq_i) < length(matchedSeq_ii)) 
        desertUAS = oldDesertUAS;
        matchedSeq_ii = matchedSeq_i;
        stuckSequenceCounter = stuckSequenceCounter + 1;
    else
        stuckSequenceCounter = 0;
    end
    if (~isempty(matchedSeq_ii)) 
        numOfMatchedSeq = length(matchedSeq_ii); 
        if (numOfMatchedSeq < minSeq)
            % ----------Stores the minimal sequence and its matches-------
            minSeq = numOfMatchedSeq;
            minSeqStr = desertUAS;
            %fprintf(fid, sprintf('%s ; MatchedSequance = %.0f\n', desertUAS, minSeq));
            if (minSeq < 20)
                play(chirpObj);  % Signals that a new minimal-matches-sequence was found
            end
        end
        % -------- Records all sequences with minimal matches ---------
        if (numOfMatchedSeq == minSeq)
            fprintf(fid, sprintf('%s ; MatchedSequance = %.0f\n', desertUAS, numOfMatchedSeq));
        end
        oldDesertUAS = desertUAS;
        desertUAS = ReplaceMatchedSeq(oldDesertUAS, matchedSeq_ii);
        matchedSeq_i = matchedSeq_ii;
        if (length(desertUAS) > seqLength)
            warning 'Current sequence is bigger than defined!';
        end
        deltaTime = toc;
        waitTime = round((loopMaxNum - stuckSequenceCounter)*deltaTime)/60;
        wbText = sprintf('Left Running Time:%.0f minutes',waitTime);
        wbText = {wbText,sprintf('Min sequance length:%.0f(%d)',minSeq,stuckSequenceCounter)};
        waitbar(stuckSequenceCounter/loopMaxNum,wb,wbText);
        continue
    end
    minSeq = length(matchedSeq_ii);
    minSeqStr = desertUAS;
    % ---------- Handles the case of perfect seqeunce (no matches) ----------
    fprintf(fid, sprintf('%s ; MatchedSequance = %.0f\n', desertUAS,minSeq));        
    fprintf(fid,'Perfect sequance was found!!!\n');
    found = 1;
    desertUAS = ManualRandSeq(seqLength);  % Generates new random sequence
    break
end