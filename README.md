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







 






