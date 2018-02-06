# Сирота Илья, 402-И

# Метод k ближайших соседей 

Постановка задачи: Задана обучающая выборка и множество классов. Требуется найти к какому классу относится классифицируемый объект. 

 Дано: xl - обучающая выборка, z - классифицируемый объект, параметр k.

Решение:
Сортируем объекты обучающей выборки xl по возрастанию расстояния от объекта z:
```R
orderedXl <- sortObjectsByDist(xl, z)
```
Далее берем только первые k элементов orderedXl: 
```R
classes <- orderedXl[1:k, n + 1]
```
Получаем список классов к которым относятся все объекты, находящиеся в массиве:
```R
classes:list <- unique(classes)
```
Далее вычисляем количество встречаемости каждого класса из k ближайщих:
```R
for (i in 1:k)
{
    counts[which(classes[i]==list)] =
      counts[which(classes[i]==list)] + 1
}
```
Класс с максимальным количеством и является классом заданного классифицируемого объекта:
```R
return (list[which.max(counts)]) 
```
![knn](https://github.com/MrAce97/smpr/blob/master/images/knn.png)


# Метод k взвешенных ближайших соседей 

Постановка задачи: Задана обучающая выборка и множество классов. Требуется найти к какому классу относится классифицируемый объект. 

 Дано: xl - обучающая выборка, z - классифицируемый объект, параметры k и q.

Решение:
Сортируем объекты обучающей выборки xl по возрастанию расстояния от объекта z:
```R
orderedXl <- sortObjectsByDist(xl, z)
```
Далее берем только первые k элементов orderedXl: 
```R
classes <- orderedXl[1:k, n + 1]
```
Получаем список классов к которым относятся все объекты, находящиеся в массиве:
```R
classes:list <- unique(classes)
```
Определим весовую функцию:
```R
w <- function(i,k,q){
  if(k>0) {
    (k+1-i)/k
  }
 else if(q>0&q<1){
    q^i
  } 
}
```
Весовая функция оценивает степень важности i-го соседа для классификации объекта z. Зависит от i обязательно, от k необязательно. 
Далее вычисляем сумму весов объектов, относящихся к одному и тому же классу:
```R
for (i in 1:k)
{
    counts[which(classes[i]==list)] =
      counts[which(classes[i]==list)] + w(i,k)
}
```
Класс с максимальным весом и является классом заданного классифицируемого объекта:
```R
return (list[which.max(counts)]) 
```
![kwnn](https://github.com/MrAce97/smpr/blob/master/images/kwnn.png)

# Метод парзеновского окна фиксированной ширины



Постановка задачи: Задана обучающая выборка и множество классов. Требуется найти к какому классу относится классифицируемый объект.

Дано: xl - обучающая выборка, z - классифицируемый объект и параметр h (ширина окна).

Решение:
Получаем список классов к которым относятся все объекты xl:
```R
list <- unique(xl[,n+1])
```
Введем убывающую функцию Kernel. Данная функция будет давать нам веса, которые будут убывать по мере роста расстояния от классифицируемого объекта. 
Парaметром функции будет расстояние от классифицируемого объекта до i-го соседа, деленного на некоторый параметр h. 
```R
kernel <- function(par){
  ifelse(abs(par)>1, 0, 3/4*(1-par^2))
}
```
Далее вычисляем сумму весов объектов, относящихся к одному и тому же классу:
```R
for (i in 1:l)
  {
    counts[which(xl[i,n+1]==list)] =
      counts[which(xl[i,n+1]==list)] + kernel(euclideanDistance(z,xl[i,1:n])/h)
  }
  ```
Класс с максимальным весом и является классом заданного классифицируемого объекта:
```R
return (list[which.max(counts)])
```

Kernel=3/4*(1-x^2)

![pwf](https://github.com/MrAce97/smpr/blob/master/images/parcenWindowFixed.png)

Kernel=2/pi/(exp(x)+exp(-x))

![pwf](https://github.com/MrAce97/smpr/blob/master/images/parsenWindowFixed1.png)


# Метод парзеновского окна переменной ширины



Постановка задачи: Задана обучающая выборка и множество классов. Требуется найти к какому классу относится классифицируемый объект.

Дано: xl - обучающая выборка, z - классифицируемый объект и параметр k.

Решение:
Сортируем объекты обучающей выборки xl по возрастанию расстояния от объекта z:
```R
orderedXl <- sortObjectsByDist(xl, z)  
```
Получаем список классов к которым относятся все объекты xl:
```R
list <- unique(xl[,n+1])
```
Как и в случае с фиксированной шириной введем функцию Kernel:
```R
kernel <- function(par){
  ifelse(abs(par)>1, 0, 3/4*(1-par^2))
}
```
Параметр h определяем, как расстояние классифицируемого объекта до k+1 соседа:
```R
h <- euclideanDistance(z,orderedXl[k+1,1:n])
```
Далее вычисляем сумму весов объектов, относящихся к одному и тому же классу:
```R
for (i in 1:l)
  {
    counts[which(xl[i,n+1]==list)] =
      counts[which(xl[i,n+1]==list)] + kernel(euclideanDistance(z,xl[i,1:n])/h)
  }
  ```
Класс с максимальным весом и является классом заданного классифицируемого объекта:
```R
return (list[which.max(counts)])
```
Kernel=3/4*(1-x^2)

![pwf](https://github.com/MrAce97/smpr/blob/master/images/parcenWindowFloat.png)

Kernel=2/pi/(exp(x)+exp(-x))

![pwf](https://github.com/MrAce97/smpr/blob/master/images/parsenWindowFloat1.png)

# Наивный байесовский классификатор

Постановка задачи: Задана обучающая выборка и классифицируемый объект. Требуется найти к какому классу относится объект.

Дано: xl - обучающая выборка, z - классифицируемый объект и параметр k.

Наивный байесовский классификатор – это алгоритм классификации, основанный на теореме Байеса с допущением о независимости признаков.

Теорема Байеса:

![tb](https://github.com/MrAce97/smpr/blob/master/images/bayes.png)

P(c|x) – апостериорная вероятность данного класса c (т.е. данного значения целевой переменной) при данном значении признака x.

P(c) – априорная вероятность данного класса.

P(x|c) – правдоподобие, т.е. вероятность данного значения признака при данном классе.

P(x) – априорная вероятность данного значения признака.

Для классифицируемого объекта вычисляются функции правдоподобия каждого из классов, по ним вычисляются апостериорные вероятности классов. Объект относится к тому классу, для которого апостериорная вероятность максимальна.

Решение:
```R
naiveBayes <- function(xl,z){
  l <- dim(xl)[1]
  n <- dim(xl)[2] - 1   
  list <- unique(xl[,n+1])
  counts = 0
  h <- c(1,1)
  apasterior <- c()
  for(i in 1:length(list)){
    temp_xl <- xl[xl$Species == list[i], ]
    apasterior_sum <-0
    for(j in 1:nrow(temp_xl)){
      apasterior_tmp<-1
      for(k in 1:n){
        apasterior_tmp <- 1/h[k]*kernel((z[k]-temp_xl[j,k])/h[k])*apasterior_tmp
      }
      apasterior_sum<-apasterior_sum+apasterior_tmp
    }
    apasterior[i]=1/nrow(temp_xl)+apasterior_sum
  }
  print(apasterior)
  counts <-c()
  for(i in 1:length(list)){
    counts[i]<-log(apasterior[i])+log(1/3)
  }
  
  return (list[which.max(counts)])
 
}

```
![naiveBayes](https://github.com/MrAce97/smpr/blob/master/images/naiveBayes.png)
# LOO

Постановка задачи: Оценить способности алгоритмов, обучаемых по выборке.

Дано: xl - обучающая выборка, a - алгоритм и параметр k.

Решение:
Убираем i-й объект с  обучающей выборки, применяем алгоритм с полученной выборкой и i-м объектом в качестве классифицируемого, получаем ответ. Если ответ совпадает 
с реальным, то продолжаем работу, то прибавляем 1 к переменной res. По окончанию работы цикла делим res на количество элементов выборки l и получаем оценку.
```R
for (i in 1:l)
  {
    if(a(xl[-i, ], xl[i,-(n+1)], k) != xl$Species[i]) 
    {
      res <- res+1
    }
  }
  return (res/l)
  ```
Недостатки:
 - Задачу обучения приходится решать N раз
 - Оценка скользящего контроля предполагает, что алгоритм обучения уже задан. Она ничего не говорит о том, какими свойствами должны обладать «хорошие» алгоритмы обучения, и как их строить.

| Алгоритм  |      LOO      |
|----------|:-------------:|
| naiveBayes |  0.05333333 |
| parcenWindowFloat |    0.09333333   |  
| parcenWindowFixed |    0.05333333   |
| kwnn |    0.04   |
| knn |    0.04   |
    

## Potential function algorithm
It's just a slight modification of parzen window algorithm:

![potential](https://github.com/toxazol/machineLearning/blob/master/img/Screenshot%20from%202017-12-16%2013-40-38.png)

Now window width *h* depends on training object *x*. *γ* represents
importance of each such object.

All these *2l* params can be obtained/adjusted with the following
procedure:

```R
potentialParamsFinder <- function(dataSet){
  n = dim(dataSet)[1]; m = dim(dataSet)[2]-1
  params = cbind(rep(1,n), rep(0,n))
  for(i in 1:n){
    repeat{
      res = potentialClassifier(dataSet, dataSet[i, 1:m], params)[1]
      if(res == dataSet[i,m+1]) break
      params[i,2] = params[i,2] + 1
    }
  }
  return(params)
}
```

Though this process on a sufficiently big sample can take considerable
amount of time. Here is the result of found and applyied params:

![potential](https://github.com/toxazol/machineLearning/blob/master/img/potential.png)

## STOLP algorithm
This algorithm implements data set compression by finding regular (etalon)
objects (stolps) and removing objects which do not or harmfully affect classification from sample.
To explain how this algorithm works idea of *margin* has to be introduced.
> **Margin** of an object (*M(x)*) shows us, how deeply current object lies within its class.
![margin](https://github.com/toxazol/machineLearning/blob/master/img/Screenshot%from%2017-12-16%15-07-53.png)

Here is **R** implementation of STOLP algorithm:
```R
STOLP <- function(set, threshold, classifier, argsList){
  plot(iris[,3:4], bg=colors2[iris$Species], col=colors2[iris$Species])
  rowsNum <- dim(set)[1]
  varsNum <- dim(set)[2]-1
  toDelete = numeric()
  for(i in 1:rowsNum){
    currentArgs = c(list(set, set[i, 1:varsNum]),argsList)
    res = do.call(classifier,currentArgs)
    if(res[1] != set[i, varsNum+1]){
      toDelete <- c(toDelete, i)
    }
  }
  points(set[toDelete,], pch=21, bg='grey', col='grey') # debug
  set = set[-toDelete, ]; rowsNum = rowsNum - length(toDelete)
  labels = levels(set[rowsNum,varsNum+1])
  maxRes = rep(0, length(labels)); names(maxRes)<-labels
  maxLabel = rep(0, length(labels)); names(maxLabel)<-labels
  for(i in 1:rowsNum){
    currentArgs = c(list(set, set[i, 1:varsNum]),argsList)
    res = do.call(classifier,currentArgs)
    if(res[2] > maxRes[res[1]]){
      maxRes[res[1]] = res[2]
      maxLabel[res[1]] = i
    }
  }
  regular = set[maxLabel, ]
  points(regular, pch=21, bg=colors2[regular$Species], col=colors2[regular$Species])
  repeat{
    errCount = 0L; toAdd = 0L; maxAbsMargin = -1
    for(i in 1:rowsNum){
      currentArgs = c(list(regular, set[i, 1:varsNum]),argsList)
      res = do.call(classifier,currentArgs)
      if(res[1] != set[i, varsNum+1]){
        errCount = errCount + 1
        if(as.double(res[2]) > maxAbsMargin)
          toAdd <- i
          maxAbsMargin <- as.double(res[2])
      }
    }
    if(errCount <= threshold)
      return(regular)
    newRegular = set[toAdd,]
    regular = rbind(regular,newRegular)
    points(newRegular, pch=21, bg=colors2[newRegular$Species], col=colors2[newRegular$Species])
  }
}
```

Here are STOLP compression results for kNN, kWNN, parzen windowss algorithms respectively:

![stolpKnn](https://github.com/toxazol/machineLearning/blob/master/img/stolpKnn.png)
![stolpKwnn](https://github.com/toxazol/machineLearning/blob/master/img/stolpKwnn.png)
![stolpParzen](https://github.com/toxazol/machineLearning/blob/master/img/stolpParzen.png)


## Conclusion

Here is a summary of results obtained by LOO for different algorithms on *iris* data set:

| algorithm                                        | best parameter | errors | empirical risk |
| -------------------------------------------------|--------------- |--------|----------------|
| kNN (4 features)                                 |      best k: 19|       3|            0.02|
| kNN (2 features)                                 |       best k: 6|       5|          0.0(3)|
|parzen window (epanechnikov kernel, 4 features)   |    best h: 0.8 |       5|          0.0(3)|
|parzen window (gaussian kernel)                   |    best h: 0.02|       5|          0.0(3)|
|parzen window /w variable window (uniform kernel) |       best k: 1|       5|          0.0(3)|
|parzen window /w variable window (gaussian kernel)|       best k: 1|       6|            0.04|
|parzen window (first 3 kernels, 2 features)       |     best h: 0.4|       6|            0.04|
| kWNN                                             |     best q: 0.6|       6|            0.04|







# Linear classification algorithms
___

## Adaline
___
ADALINE stands for **Ada**ptive **Lin**ear **E**lement. It was developed by Professor Bernard Widrow and his graduate student Ted Hoff at Stanford University in 1960. It is based on the McCulloch-Pitts model and consists of a weight, a bias and a summation function.

Operation: ![adaline](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/adaline_1.svg)

Its adaptation is defined through a cost function (error metric) of the residual ![adaline](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/adaline_2.svg) where ![adaline](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/adaline_3.svg) is the desired input. With the MSE error metric ![adaline](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/adaline_4.svg) adapted weight and bias become: ![adaline](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/adaline_5.svg) and ![adaline](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/adaline_6.svg)

The Adaline has practical applications in the controls area. A single neuron with tap delayed inputs (the number of inputs is bounded by the lowest frequency present and the Nyquist rate) can be used to determine the higher order transfer function of a physical system via the bi-linear z-transform. This is done as the Adaline is, functionally, an adaptive FIR filter. Like the single-layer perceptron, ADALINE has a counterpart in statistical modelling, in this case least squares regression.

![Adaline](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/ADALINE.png)
```R
## Квадратичная функция потерь
lossQuad <- function(x)
{
  return ((x-1)^2)
}
## Стохастический градиент для ADALINE
sg.ADALINE <- function(xl, eta = 1, lambda = 1/6)
{
	l <- dim(xl)[1]
	n <- dim(xl)[2] - 1
	w <- c(1/2, 1/2, 1/2)
	iterCount <- 0
	## initialize Q
	Q <- 0
	for (i in 1:l)
	{
	  ## calculate the scalar product <w,x>
	  wx <- sum(w * xl[i, 1:n])
	  ## calculate a margin
	  margin <- wx * xl[i, n + 1]
	  Q <- Q + lossQuad(margin)
	}
	repeat
	{
	  ## calculate the margins for all objects of the training sample
	  margins <- array(dim = l)
	 
	  for (i in 1:l)
	  {
		xi <- xl[i, 1:n]
		yi <- xl[i, n + 1]
		margins[i] <- crossprod(w, xi) * yi
	  }
	  ## select the error objects
	  errorIndexes <- which(margins <= 0)
	  if (length(errorIndexes) > 0)
	  {
		# select the random index from the errors
		i <- sample(errorIndexes, 1)
		iterCount <- iterCount + 1
		xi <- xl[i, 1:n]
		yi <- xl[i, n + 1]
		## calculate the scalar product <w,xi>
		wx <- sum(w * xi)
		## make a gradient step
		margin <- wx * yi
		## calculate an error
		ex <- lossQuad(margin)
		eta <- 1 / sqrt(sum(xi * xi))
		w <- w - eta * (wx - yi) * xi
		## Calculate a new Q
		Qprev <- Q
		Q <- (1 - lambda) * Q + lambda * ex
	  }
	  else
	  {
		break
	  }
	}
	return (w)
}
```

## Hebb
___
From the point of view of artificial neurons and artificial neural networks, Hebb's principle can be described as a method of determining how to alter the weights between model neurons. The weight between two neurons increases if the two neurons activate simultaneously, and reduces if they activate separately. Nodes that tend to be either both positive or both negative at the same time have strong positive weights, while those that tend to be opposite have strong negative weights.

The following is a formulaic description of Hebbian learning:

![hebb](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/hebb_1.svg)

Another formulaic description is:

![hebb](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/hebb_2.svg)

Hebb's Rule is often generalized as ![hebb](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/hebb_3.svg) ![hebb](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/hebb_4.svg)

![Hebb](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/Hebbs_rule.png)
```R
## Функция потерь для правила Хэбба
lossPerceptron <- function(x)
{
	return (max(-x, 0))
}
## Стохастический градиент с правилом Хебба
sg.Hebb <- function(xl, eta = 0.1, lambda = 1/6)
{
	l <- dim(xl)[1]
	n <- dim(xl)[2] - 1
	w <- c(1/2, 1/2, 1/2)
	iterCount <- 0
	## initialize Q
	Q <- 0
	for (i in 1:l)
	{
		## calculate the scalar product <w,x>
		wx <- sum(w * xl[i, 1:n])
		## calculate a margin
		margin <- wx * xl[i, n + 1]
		# Q <- Q + lossQuad(margin)
		Q <- Q + lossPerceptron(margin)
	}
	repeat
	{
		## Поиск ошибочных объектов
		margins <- array(dim = l)
		for (i in 1:l)
		{
			xi <- xl[i, 1:n]
			yi <- xl[i, n + 1]
			margins[i] <- crossprod(w, xi) * yi
		}
		## выбрать ошибочные объекты
		errorIndexes <- which(margins <= 0)
		if (length(errorIndexes) > 0)
		{
			# выбрать случайный ошибочный объект
			i <- sample(errorIndexes, 1)
			iterCount <- iterCount + 1
			xi <- xl[i, 1:n]
			yi <- xl[i, n + 1]
			w <- w + eta * yi * xi
		}
		else
		break;
	}
	return (w)
}
```



## Plug-In
___
Estimate the parameters of the likelihood functions µˆy and Σy acting on the parts of the training samples xl(y) for each class y in Y. Then the sample of evaluations we substitute in the optimal Bayesian classifier. Get a normal Bayesian classifier, which is also known as a _**a wildcard (plug-in)**_.

In the asymptotic expansion of l(y) -> oo assessment µˆy and Σy acting have several optimal properties: they are _*not*displaced, efficient and solvent**_. However, estimates made on short samples may not be accurate enough.

![plug-in](images/plug-in_quadro.png)
```R
## Получение коэффициентов подстановочного алгоритма
getPlugInDiskriminantCoeffs <- function(mu1, sigma1, mu2, sigma2)
{
  ## Line equation: a*x1^2 + b*x1*x2 + c*x2 + d*x1 + e*x2 + f = 0
  invSigma1 <- solve(sigma1)
  invSigma2 <- solve(sigma2)
  f <- log(abs(det(sigma1))) - log(abs(det(sigma2))) +
    mu1 %*% invSigma1 %*% t(mu1) - mu2 %*% invSigma2 %*%
    t(mu2);
  alpha <- invSigma1 - invSigma2
  a <- alpha[1, 1]
  b <- 2 * alpha[1, 2]
  c <- alpha[2, 2]
  beta <- invSigma1 %*% t(mu1) - invSigma2 %*% t(mu2)
  d <- -2 * beta[1, 1]
  e <- -2 * beta[2, 1]
  return (c("x^2" = a, "xy" = b, "y^2" = c, "x" = d, "y"
            = e, "1" = f))
}
```

## LDF
___
Let the covariance matrices of the classes are the same and equal to Σ.

![sigma](images/LDF_sigma.gif)

In this case, the separating surface is piecewise linear. The wildcard algorithm is:

![a](images/LDF_a.gif)

![LDF](images/LDF.png)
```R
## Оценка ковариационной матрицы для ЛДФ
estimateFisherCovarianceMatrix <- function(objects1, objects2, mu1, mu2)
{
	rows1 <- dim(objects1)[1]
	rows2 <- dim(objects2)[1]
	rows <- rows1 + rows2
	cols <- dim(objects1)[2]
	sigma <- matrix(0, cols, cols)
	for (i in 1:rows1)
	{
		sigma <- sigma + (t(objects1[i,] - mu1) %*%
		(objects1[i,] - mu1)) / (rows + 2)
	}
	for (i in 1:rows2)
	{
		sigma <- sigma + (t(objects2[i,] - mu2) %*%
		(objects2[i,] - mu2)) / (rows + 2)
	}
	return (sigma)
}
```
## Naive Bayesian Classifier (NBC)
___
Below is the Naive Bayes’ Theorem:

P(A | B) = P(A) * P(B | A) / P(B)

Which can be derived from the general multiplication formula for AND events:

P(A and B) = P(A) * P(B | A)

P(B | A) = P(A and B) / P(A)

P(B | A) = P(B) * P(A | B) / P(A)

If I replace the letters with meaningful words as I have been adopting throughout, the Naive Bayes formula becomes:

P(outcome | evidence) = P(outcome) * P(evidence | outcome) / P(evidence)

It is with this formula that the Naive Bayes classifier calculates conditional probabilities for a class outcome given prior information.

The reason it is termed “naive” is because we assume independence between attributes when in reality they may be dependent in some way.

So let`s try to implement the naive Bayesian classifier and see what we get.
Before you use the source below you need install several packages & load some libraries...

```R
install.packages("caret")
install.packages("MASS")
install.packages("klaR")
require("caret")
require(lattice)
require(ggplot2)
require(klaR)
require(MASS)
require(e1071) #predict
```

Loading data & Creating the model

```R
data("iris")
model <- NaiveBayes(Species ~ ., data = iris)
```

**predict** computes the conditional a-posterior probabilities of a categorical class variable given independent predictor variables using the Bayes rule.

```R
preds <- predict(model, iris[,-5])
```

Here they are

```R
> preds$posterior
              setosa   versicolor    virginica
  [1,]  1.000000e+00 2.981309e-18 2.152373e-25
  [2,]  1.000000e+00 3.169312e-17 6.938030e-25
  [3,]  1.000000e+00 2.367113e-18 7.240956e-26
  ...
[149,] 1.439996e-195 3.384156e-07 9.999997e-01
[150,] 2.771480e-143 5.987903e-02 9.401210e-01
```

The last one, we need to know how many error classified, so we need to compare the result of prediction with the class/iris species.

```R
table(predict(model, iris[,-5])$class, iris[,5])

##             y
##              setosa versicolor virginica
##   setosa         50          0         0
##   versicolor      0         47         3
##   virginica       0          3        47
```

If you want to plot the features with Naive Bayes, you can use this command:

```R
naive_iris <- NaiveBayes(iris$Species ~ ., data = iris)
plot(naive_iris)
```

![b_iris_PL](https://github.com/naemnamenmea/SMCS/blob/master/Baessian/images/b_iris_PL.png)
![b_iris_PW](https://github.com/naemnamenmea/SMCS/blob/master/Baessian/images/b_iris_PW.png)
![b_iris_SL](https://github.com/naemnamenmea/SMCS/blob/master/Baessian/images/b_iris_SL.png)
![b_iris_SW](https://github.com/naemnamenmea/SMCS/blob/master/Baessian/images/b_iris_SW.png)

## Plug-In
___
Estimate the parameters of the likelihood functions µˆy and Σy acting on the parts of the training samples xl(y) for each class y in Y. Then the sample of evaluations we substitute in the optimal Bayesian classifier. Get a normal Bayesian classifier, which is also known as a _**a wildcard (plug-in)**_.

In the asymptotic expansion of l(y) -> oo assessment µˆy and Σy acting have several optimal properties: they are _*not*displaced, efficient and solvent**_. However, estimates made on short samples may not be accurate enough.

![plug-in](https://github.com/naemnamenmea/SMCS/blob/master/Baessian/images/plug-in_quadro.png)
```R
## Получение коэффициентов подстановочного алгоритма
getPlugInDiskriminantCoeffs <- function(mu1, sigma1, mu2, sigma2)
{
  ## Line equation: a*x1^2 + b*x1*x2 + c*x2 + d*x1 + e*x2 + f = 0
  invSigma1 <- solve(sigma1)
  invSigma2 <- solve(sigma2)
  f <- log(abs(det(sigma1))) - log(abs(det(sigma2))) +
    mu1 %*% invSigma1 %*% t(mu1) - mu2 %*% invSigma2 %*%
    t(mu2);
  alpha <- invSigma1 - invSigma2
  a <- alpha[1, 1]
  b <- 2 * alpha[1, 2]
  c <- alpha[2, 2]
  beta <- invSigma1 %*% t(mu1) - invSigma2 %*% t(mu2)
  d <- -2 * beta[1, 1]
  e <- -2 * beta[2, 1]
  return (c("x^2" = a, "xy" = b, "y^2" = c, "x" = d, "y"
            = e, "1" = f))
}
```

## LDF
___
Let the covariance matrices of the classes are the same and equal to Σ.

![sigma](https://github.com/naemnamenmea/SMCS/blob/master/Baessian/images/LDF_sigma.gif)

In this case, the separating surface is piecewise linear. The wildcard algorithm is:

![a](https://github.com/naemnamenmea/SMCS/blob/master/Baessian/images/LDF_a.gif)

![LDF](https://github.com/naemnamenmea/SMCS/blob/master/Baessian/images/LDF.png)
```R
## Оценка ковариационной матрицы для ЛДФ
estimateFisherCovarianceMatrix <- function(objects1, objects2, mu1, mu2)
{
	rows1 <- dim(objects1)[1]
	rows2 <- dim(objects2)[1]
	rows <- rows1 + rows2
	cols <- dim(objects1)[2]
	sigma <- matrix(0, cols, cols)
	for (i in 1:rows1)
	{
		sigma <- sigma + (t(objects1[i,] - mu1) %*%
		(objects1[i,] - mu1)) / (rows + 2)
	}
	for (i in 1:rows2)
	{
		sigma <- sigma + (t(objects2[i,] - mu2) %*%
		(objects2[i,] - mu2)) / (rows + 2)
	}
	return (sigma)
}
```
