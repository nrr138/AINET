function [hurst] = hurstf(data0)
data=data0;  % Local copy
[M,npoints]=size(data0);
yvals=zeros(1,npoints);
xvals=zeros(1,npoints);
data2=zeros(1,npoints);
index=0;
binsize=1;
while npoints>4
  
    y=std(data);
    index=index+1;
    xvals(index)=binsize;
    yvals(index)=binsize*y;
    
    npoints=fix(npoints/2);
    binsize=binsize*2;
    for ipoints=1:npoints
        data2(ipoints)=(data(2*ipoints)+data((2*ipoints)-1))*0.5;
    end
    data=data2(1:npoints);
    
end

xvals=xvals(1:index);
yvals=yvals(1:index);

logx=log(xvals);
logy=log(yvals);

p2=polyfit(logx,logy,1);

hurst=p2(1); % Hurst exponent = the slope of the linear fit of the log-log plot
return;