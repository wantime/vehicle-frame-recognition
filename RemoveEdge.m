function [ f ] = RemoveEdge( img )
%UNTITLED5 È¥³ý±ßÔµµÄ°×±ß
% 
    [m,n] = size(img);
    for i = 1:m
        for j = 1:n
            if((i < 10)||(j < 10)||(i>m-7)||(j>n-7))
                img(i,j) = 0;
            end
        end
    end
    f = img;
    %imshow(f);
end

