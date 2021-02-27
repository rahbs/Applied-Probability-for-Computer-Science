# Applied-Probability-for-Computer-Science
##  2020-2 Applied Probability for Computer Science
##  컴퓨터 응용 확률(이상철 교수님) 수업 과제 정리

### hw3 : regression of exponential model
-	지수적으로 증가 혹은 감소하는 데이터를 모델링 시, 데이터에 자연로그를 취하여 선형 모델로 회귀분석

<p align="center"><img src="https://render.githubusercontent.com/render/math?math=y=a\times e^{bx}"></p>

Parameter a와 b를 찾기 위해, 다음과 같은 계산 과정을 거쳤다.

<p align="center"><img src="https://render.githubusercontent.com/render/math?math=y=ae^{bx}"></p>

<p align="center"><img src="https://render.githubusercontent.com/render/math?math=\ln{y}=\ln{\left(ae^{bx}\right)}=\ln{a} %2B bx"></p>

<p align="center">Let <img src="https://render.githubusercontent.com/render/math?math=\ \ln{y}=z,\ln{a}=a_0,b=a_1,"></p>

<p align="center">then <img src="https://render.githubusercontent.com/render/math?math=z=a_0 %2B a_1x  ">
(linear)</p>

위와 같은 변환을 통해, non-linear 모델을 linear model 로 변환하였고, 
변환된 linear model의 parameter인 <img src="https://render.githubusercontent.com/render/math?math=a_0 , a_1">을 찾고자 한다.
모델의 값과, 실제 값의 차이를 나타내는 SSE를 다음과 같이 정의하자.
<p align="center"><img src="https://render.githubusercontent.com/render/math?math=SSE=\ \sum\ \ {(\ z_i-\ \widehat{z_i})}^2\  = =\ \sum\ \ {(\ z_i-\ a_0-a_1x_i)}^2"></p>

SSE를 최소화하는 <img src="https://render.githubusercontent.com/render/math?math=a_0 , a_1"> 값을 찾기 위해, 다음과 같은 미분과정을 거친다.

<p align="center"><img src="https://render.githubusercontent.com/render/math?math=\frac{\partial SSE\ }{\partial a_0} = 2\sum\left(\ z_i-\ a_0-a_1x_i\ \right)\left(-1\right)=0\ "></p>
			
<p align="center"><img src="https://render.githubusercontent.com/render/math?math=\frac{\partial SSE\ }{\partial a_1} = 2\sum\left(\ z_i-\ a_0-a_1x_i\ \right)\left(-x_i\right) = 0"></p>

계산을 통해 아래와 다음과 같은 식을 얻어낼 수 있다.

<p align="center"><img src="https://render.githubusercontent.com/render/math?math=\large a_1\ =\frac{\ n\sum_{i=1}^{n}{z_ix_i}-\ \sum_{i=1}^{n}z_i\sum_{i=1}^{n}x_i}{n\sum_{i=1}^{n}{x_i}^2-\left(\sum_{i=1}^{n}x_i\right)^2}"></p>

<p align="center"><img src="https://render.githubusercontent.com/render/math?math=a_0\ =\overline{z}\ -a_1\overline{x}">    ( <img src="https://render.githubusercontent.com/render/math?math=\overline{z}"> ∶ z 값의 평균, <img src="https://render.githubusercontent.com/render/math?math=\overline{x}"> : x 값의 평균) </p>

                  
<p align="center"><img src="https://render.githubusercontent.com/render/math?math=a\ =e^{a_0}, b\ = a_1"></p>

- original data (exponential)

![qq](https://user-images.githubusercontent.com/35826556/109385047-46526f80-7934-11eb-84c9-1abe3b0ea632.png)

- transformed data (linear)

![ww](https://user-images.githubusercontent.com/35826556/109385064-713cc380-7934-11eb-90f5-618da86f157b.png)

### hw4 : iris classfication
- 

### hw5 : binomial random variable vs poisson distribution
