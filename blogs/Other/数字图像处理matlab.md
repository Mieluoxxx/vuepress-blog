---
title: 数字图像处理matlab
date: 2023/10/31 00:30:00
categories:
    - 计算机视觉
tags: 
    - 数字图像处理
    - matlab
---

## 图像处理初步

### 图像的读取，显示和存储，以及提取图像的基本信息

```matlab
% 图像的读取
lena = imread("lena.bmp");
lena_RGB = imread('lenaRGB.bmp');
% 提取图像的基本信息
whos lena;
% 图像的显示
figure;
imshow(lena);
% 图像的保存
imwrite(lena, 'lena2.png');
% 观察两幅图像的质量
figure;
subplot(121);imshow(lena);
subplot(122);imshow(lena_RGB);
```


### 图像间的代数运算

```matlab
% 加法
img1 = imresize(imread("lenaRGB.bmp"),[268 216]);
img2 = imread("pout.jpg");
img3 = img1 + img2;
img4 = imadd(img1,img2);
figure;
subplot(141);imshow(img1);title('lena原始图像');
subplot(142);imshow(img2);title('pout原始图像');
subplot(143);imshow(img3);title('相加后图像');
subplot(144);imshow(img4);title('imadd后图像');

%减法
img1 = imresize(imread("lenaRGB.bmp"),[268 216]);
img2 = imread("pout.jpg");
img3 = img1 - img2;
img4 = imsubtract(img1,img2);
figure;
subplot(141);imshow(img1);title('lena原始图像');
subplot(142);imshow(img2);title('pout原始图像');
subplot(143);imshow(img3);title('相减后图像');
subplot(144);imshow(img4);title('imsubtract后图像');

%乘法
img1 = imresize(imread("lenaRGB.bmp"),[268 216]);
img2 = imread("pout.jpg");
img3 = uint16(img1).*uint16(img2);
img4 = immultiply(uint16(img1),uint16(img2));
figure;
subplot(141);imshow(img1);title('lena原始图像');
subplot(142);imshow(img2);title('pout原始图像');
subplot(143);imshow(img3);title('相乘后图像')
subplot(144);imshow(img4);title('immultiply后图像');

%除法
img1 = imresize(imread("lenaRGB.bmp"),[268 216]);
img2 = imread("pout.jpg");
img3 = double(img1)./double(img2);
img4 = imdivide(double(img1),double(img2));
figure;
subplot(141);imshow(img1);title('lena原始图像');
subplot(142);imshow(img2);title('pout原始图像');
subplot(143);imshow(img3);title('相除后图像')
subplot(144);imshow(img4);title('immultiply后图像');
```



### 图像的线性运算

```matlab
g = imread("cameraman.bmp");
f1 = 1.2 * g;
figure;
subplot(121);imshow(g);title('原始图像');
subplot(122);imshow(f1);title('线性运算图像');
saveas(gcf, 'result6.png')

img1 = imread("cameraman.bmp");
img2 = immultiply(img1,1.2);
figure;
subplot(121);imshow(img1);title('原始图像');
subplot(122);imshow(img2);title('缩放运算图像');
saveas(gcf, 'result7.png')
```



## 图像增强

### 图像的求反
```matlab
% 取反
lena = imread('lena.bmp');
lena_r = 255 - lena;
figure;
subplot(121);imshow(lena);title('原始图像');
subplot(122);imshow(lena_r);title('求反图像');
saveas(gcf,'result1.png');
```



### 线性灰度变换

```matlab
% 线性灰度变换
lena = imread('lena.bmp');
[M,N]=size(lena);
for i=1:M
    for j=1:N
        if lena(i,j)<150
            lena(i,j)=(200-30)/(150-30)*(lena(i,j)-30)+30;
        elseif lena(i,j) >= 150 && lena(i,j) <= 255
            lena(i,j)=(255-200)/(255-150)*(lena(i,j)-150)+200;
        end
    end
end
figure;
subplot(121);imshow(lena);title('原始图像');
subplot(122);imshow(lena);title('线性灰度变换图像');
saveas(gcf,'result2.png');
```



### 直方图均衡化

```matlab
% 直方图均衡化
lena = imread('lena.bmp');
lena_h = histeq(lena);
figure;
subplot(2,2,1);imshow(lena);title('原始图像');
subplot(2,2,2);imshow(lena_h);title('均衡化后图像');
subplot(2,2,3);imhist(lena);title('图像直方图');
subplot(2,2,4);imhist(lena_h);title('直方图均衡化');
saveas(gcf,'result3.png');
```



### 平滑滤波器（模糊处理和减少噪声）

```matlab
% 平滑滤波器
lena = imread('lena.bmp');
lena_n = imnoise(lena,'salt & pepper',0.02);
figure;
subplot(1,2,1),imshow(lena),title('原始图像');
subplot(1,2,2),imshow(lena_n),title('加入噪声密度：0.02的椒盐噪声');
saveas(gcf,'result4.png');

lena3 = filter2(fspecial('average',3),lena_n)/255; 
lena5 = filter2(fspecial('average',5),lena_n)/255; 
figure;
subplot(1,3,1),imshow(lena_n),title('0.02的椒盐噪声图像');
subplot(1,3,2),imshow(lena3),title('3*3模板平滑滤波');
subplot(1,3,3),imshow(lena5),title('5*5模板平滑滤波');
saveas(gcf,'result5.png');
```



### 锐化滤波器（使边缘和轮廓线清晰）

```matlab
% 锐化滤波器
lena = imread('lena.bmp');
w = [1, 1, 1; 1, -8, 1; 1, 1, 1];
J = conv2(double(lena), w);
figure;
subplot(121);imshow(lena);title('原始图像');
subplot(122);imshow(uint8(J));title('锐化后的图像');
saveas(gcf,'result6.png');
```

