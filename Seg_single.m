img_path = 'E:\code\matlab\车牌\getword\data\3.png'; 
img=imread(img_path); %Img=imread('30test.png')
    %2.对图像进行阈值分割得到二值图像
   %img = region_yuzhi(img);
   %3.区域增长法去噪
   %img = RG(img);
    %4.除去图像边缘的黑边
  img = RemoveEdge(img);

    %5.获取该图片每一列的合
    col=sum(img); 
   %6.切割字符由三部分组成
   %图片的宽度
   [height,width] = size(img);
   %起始位置left
   left = 0;
   %从起始位置向终止为止移动的指针index,该值在0.85个字符宽度与1.15个字符宽度之间
   index = 1;%0.85*40=34，1.15*40=46
    %字符宽度
    lw = 40;
    %采用 cell结构来存储切割的字符 
    Letter = cell(1,17);
    %seg数组存储每个字符的起始与终止位置
    seg = zeros(2,17);

for i = 1:17;    
index = 1; 
        %指针小于两个字符的宽度
        %寻找字符的左边界，
         while col(left + index)==0 
             %记录第一列像元和值不为0的列数
             index = index+1;  
         end
         %非0的前一个位置
         left = left + index - 1; %字符分割的左标志列
         %找到左边界后指针重置
         %解决字符串断裂的情况
         index = 20;
         %判断在一个字符宽度内col为0的位置，并且指针不能超出图片宽度
         while ((col(left+index))~=0) && (index<=2*lw-1) && ((left + index)<width) %如果在有噪声的情况下如何处理？是否应该设定一个阈值？
             index = index+1;
         end
         

         %判断是否发生字符粘连在一起了
         %如果字符宽度大于1.5倍的字符，就认为是粘连
         if (index >= 1.2*lw)
             %这里需要求出left:left+index之间最小值的位置
            [val,num] = min(col(1,left+lw*0.85:left+lw*1.15));   
            % Lw =num - Left_index; %计算当前字母的实际宽度，并作为下一个字母的宽度的参考值。 %根据实际情况看是否采用该策略，还是取决于噪声的多少。
            index = num + lw*0.85;
         end
          Letter{i} =  img(:,left:left+index); %将字符切割并保存
          %seg(1,i) = left;
          left = left + index;
          %seg(2,i) = left;
          subplot(5,4,i),imshow(Letter{i});
end