function Findm=findMin(C)
  N=length(C);
for j=1:N-1
    if C(j)>=C(j+1)
        q=C(j);
    else
        Findm=j;
        break;
    end
%     continue
end
end
        
