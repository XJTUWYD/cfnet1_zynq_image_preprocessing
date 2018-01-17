# cfnet1_zynq_image_preprocessing
when we want to use fpga to implement the speed up of the cfnet1, we plan to use arm to realize the image processing 

图像预处理：

每次处理一个视频流，已知第一帧图像的groundtruth，groundtruth包括四个元素(xmin,ymin,w,h)。(targetsize =(w,h),targetposition=(xmin+w/2,ymin+h/2))

获取图像RGB每个channel的像素平均值，作为后面图像padding的准备。

avgChans = gather([mean(mean(im(:,:,1))) mean(mean(im(:,:,2))) mean(mean(im(:,:,3)))]);

instanceSize = 255 第t帧图像预处理之后的大小

exemplarSize =255 第t-1帧图像预处理之后的大小

numScale = 3 搜索第t帧数据时的尺度空间的数目，如果有三个尺度空间，第一个尺度空间会缩小图像，第二个尺度空间图像大小不变，第三个尺度空间扩大图像。

scaleStep = 1.03

wc_z = targetsize(2) + contextAmount*sum(p.targetSize);

hc_z = targetSize(1) + contextAmount*sum(p.targetSize);

s_z = sqrt(wc_z*hc_z);

s_x = instanceSize/exemplarSize * s_z;

scales = (p.scaleStep .^ ((ceil(p.numScale/2)-p.numScale) : floor(p.numScale/2)));

scaledExemplar = s_z .* scales;  1x3double

exemplar的初始化

inside_scale = round(scaledExemplar);

search_side = exemplarSize*(max(inside_scale)/min(inside_scale))

以targetposition为中心将max(inside_scale)的数据裁剪下来，如果targetposition在图像的边缘，则使用前面计算好的avgChans补全。然后将图像imresize成search_side大小变成image_1。

Target_side = round(exemplarSize* inside_scale(2)/inside_scale(1))

以image_1的中心为中心，先将图像剪切成Target_side大小最终resize成为exemplarSize大小。

设置一个汉宁惩罚窗

Window = han(scoresize*responseup)*han(scoresize*responseup)’

归一化

Window = window/sum(window(:))

为了给s_z和s_x设置最大值最小值。minfactor = 0.2，maxfactor = 5;

Min_s_x = minfactor*s_x;
Max_s_x = maxfactor*s_x;
Min_s_z = minfactor*s_z;
Max_s_z = maxfactor*s_z;

接下来就是处理需要寻找boundingbox的第t帧的数据，将其扩大缩小形成三个尺度空间。

scaleInstance = s_x*scales;

inside_scale = round(scaleInstance);

search_side = instanceSize *(max(inside_scale)/min(inside_scale))

以targetposition为中心将max(inside_scale)的数据裁剪下来，如果targetposition在图像的边缘，则使用前面计算好的avgChans补全。然后将图像imresize成search_side大小变成image_2。

Target_side_1 = round(instanceSize * inside_scale(1)/inside_scale(1))
Target_side_2 = round(instanceSize * inside_scale(2)/inside_scale(1))
Target_side_3 = round(instanceSize * inside_scale(3)/inside_scale(1))

以image_2的中心为中心，先将图像剪切成Target_side_1大小最终resize成为instanceSize大小作为第一个尺度空间。以image_2的中心为中心，先将图像剪切成Target_side_2大小最终resize成为instanceSize大小作为第二个尺度空间。以image_2的中心为中心，先将图像剪切成Target_side_3大小最终resize成为instanceSize大小作为第三个尺度空间。

最终的处理结果：


![original image](https://github.com/XJTUWYD/cfnet1_zynq_image_preprocessing/tree/master/img/1.png)

![exemplar_image](https://github.com/XJTUWYD/cfnet1_zynq_image_preprocessing/tree/master/img/2.png)

![x_crops](https://github.com/XJTUWYD/cfnet1_zynq_image_preprocessing/tree/master/img/3.png)
