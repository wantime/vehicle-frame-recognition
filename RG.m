function [ K ] = RG( I )
%RG 区域增长法去除细小噪声
    I = im2double(I); %图像灰度值归一化到[0,1]之间
    [h,w] = size(I); 
    J = zeros(size(I));%用于判断点是否已被访问过的图片
    K = J;%用于复制生长区域的图像
    Isizes = size(I);
    neg_free = 10000; %动态分配内存的时候每次申请的连续空间大小
    neg_list = zeros(neg_free,2);%定义邻域列表，并且预先分配用于储存待分析的像素点的坐标值和灰度值的空间，加速
    %如果图像比较大，需要结合neg_free来实现matlab内存的动态分配
    point = 0;%用于记录有效像素点的个数
    index = 0;
    list = neg_list;
    neigb = [ -1 0;
              1  0;
              0 -1;
              0  1];
    for m =1:w%对整幅图片进行扫描，先固定宽度，也就是逐列扫描
        for n = 1:h
            x = n;
            y = m;
            while (I(x,y) == 1)
            %增加新的邻域像素到neg_list中
                J(x,y) = 1;
                for j=1:4    %这里是访问四领域
                    xn = x + neigb(j,1);
                    yn = y + neigb(j,2);
                    ins = (xn>=1)&&(yn>=1)&&(xn<=h)&&(yn<=w);%检查邻域像素是否超过了图像的边界

                    if( ins && J(xn,yn) == 0 && I(xn,yn) == 1) %如果邻域像素在图像内部，并且像元值有效；那么将它添加到邻域列表中
                         point = point + 1;
                         index = index + 1;
                         neg_list(index,:) = [xn, yn];%存储符合要求的领域坐标
                         list(point,:) = [xn, yn];
                         J(xn,yn) = 1;%标注该邻域像素点已经被访问过 并不意味着，他在分割区域内
                    end
                end
                    %指定新的种子点
                if(index == 0)
                    break;
                end
                x = neg_list(index,1);
                y = neg_list(index,2);
                    %将新的种子点从待分析的邻域像素列表中移除
                index = index -1;
            end

            if(point > 40)    %如果有效像元的数值超过10个，那么就认为找到了联通的目标区域，开始进行赋值
                for i = 1:point
                         %标志该像素点已经是分割好的像素点
                         x = list(i,1);
                         y = list(i,2);
                         K(x,y)=1;
                end        
            end
                        point = 0;
                        neg_list = zeros(neg_free,2);
                        list = neg_list;
       end
    end
end

