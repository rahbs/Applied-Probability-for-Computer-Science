# Applied-Probability-for-Computer-Science
##  2020-2 Applied Probability for Computer Science
##  컴퓨터 응용 확률 - 이상철 교수님 수업 과제 정리 (자세한 설명/결과는 각 파일의 report 참고)

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
#### k-fold cross validation: 
각 fold의 train, validation data의 클래스 비율을 같게 총 5개의 fold로 나누었다.
총 데이터 개수: 150개
fold별 train data 개수: 120개 = (각 클래스별 40개) * (클래스 개수 3개)
fold별 validation data 개수: 30개 = (각 클래스별 10개) * (클래스 개수 3개)

#### classifier
1. Naïve bayesian classifier:
<p align="center"><img src="https://render.githubusercontent.com/render/math?math=P\left(c| x\right)=P\left(x| w_j\right)\ P\left(w_j\right)\ /\ P(x)"> --- (식 1)</p>

<p align="center">(Posterior) = (likelihood)*(prior)/(evidence)</p>

  - Likelihood: 
<p align="center"><img src="https://render.githubusercontent.com/render/math?math=P\left(x| w_j\right)=P\left(x_1| w_j\right)P\left(x_2| w_j\right)P\left(x_3| w_j\right)P\left(x_4| w_j\right)"> --- (식 2)</p>

총 4개의 feature에 대해 각각의 pmf를 구하고, 각 feature는 독립이라고 가정하여 각 값들을 곱하였다.

  - Pmf: 
	각 fold의 train data를이용한다. 
	Data를 class, feature별로 구분한 뒤, interval로 나누어 그 구간에 대한 likelihood값을 구함. 


2. Gaussian classifier:
 
 각 feature별로 likelihood를 구해서 곱하는 방식은 위와 같으나, likelihood를 구할 때 pmf를 이용하는 것이 아니라, gaussian으로 분포를 가정하여 구하였다. 

3. multi-variate Gaussian classifier

![image](https://user-images.githubusercontent.com/35826556/109386355-88cc7a00-793d-11eb-8040-34ff3086b3d2.png)

4. simple deep neural network

#### Evaluation
각 fold에 대하여, validation data의 각 행마다 Bayesian_classifier를 통해 inference를 진행 후, 실제 class값과 비교하여 precision 과 recall을 구했다.
최종적으로 각 fold의 precision과 recall의 평균값을 구해 모델을 평가하였다.


### hw5 : binomial random variable vs poisson distribution

- binomial random variable (p = 0.01 , n = 1000)

![q](https://user-images.githubusercontent.com/35826556/109385931-9c2a1600-793a-11eb-8133-1d2f1c40ffb9.png)

- pmf derived from poisson approximation (<img src="https://render.githubusercontent.com/render/math?math=\lambda">  = 10 )
![www](https://user-images.githubusercontent.com/35826556/109385934-9df3d980-793a-11eb-90c3-4225e85e0844.png)
