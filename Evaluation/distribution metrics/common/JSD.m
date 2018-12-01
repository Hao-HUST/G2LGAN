function [ dist ] = JSD( P, Q )
    if size(P,2)~=size(Q,2)
        error('the number of columns in P and Q should be the same');
    end
    M = 0.5.*(P + Q);
    dist = 0.5.*KLDiv(P,M) + 0.5*KLDiv(Q,M);
end


function dist=KLDiv(P,Q)
    if size(P,2)~=size(Q,2)
        error('the number of columns in P and Q should be the same');
    end

    if sum(~isfinite(P(:))) + sum(~isfinite(Q(:)))
       error('the inputs contain non-finite values!') 
    end
    
    dist =  sum(P.*log2(P./Q),1);

% resolving the case when P(i)==0
    dist(isnan(dist))=0;
    dist = sum(dist,2);
end
