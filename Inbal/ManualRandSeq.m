function [ seq ] = ManualRandSeq( len )
seq = [];
randomBase = [];
for i=1:len
	tempBase = randi([1,4],1);
    if (tempBase==1)
        randomBase = 'A';
    elseif (tempBase==2)
        randomBase = 'T';
    elseif (tempBase==3)
        randomBase = 'G';
    elseif (tempBase==4)
        randomBase = 'C';
    end
    seq = [seq, randomBase]   ;    
end

end

