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
    


  ## Parzen window algorithm
Let' s define *ω(i, u)* as a function of distance rather than neighbor rank.

![parzenw](https://github.com/toxazol/machineLearning/blob/master/img/Screenshot%20from%202017-12-16%2013-27-53.png?raw=true) 

where *K(z)* is  nonincreasing on [0, ∞)  kernel function. Then our metric classifier will look like this:

![parzen](https://github.com/toxazol/machineLearning/blob/master/img/Screenshot%20from%202017-12-16%2013-31-51.png?raw=true)

This is how kernel functions graphs look like:
![kernels](https://github.com/toxazol/machineLearning/blob/master/img/main-qimg-ece54bb2db23a4f823e3fdb6058761e8.png?raw=true)

Here are some of them implemented in *R*:
```R
ker1 <- (function(r) max(0.75*(1-r*r),0)) # epanechnikov
ker2 <- (function(r) max(0,9375*(1-r*r),0)) # quartic
ker3 <- (function(r) max(1-abs(r),0)) # triangle
ker4 <- (function(r) ((2*pi)^(-0.5))*exp(-0.5*r*r)) # gaussian
ker5 <- (function(r) ifelse(abs(r)<=1, 0.5, 0)) # uniform
```

Parameter *h* is called "window width" and is similar to number of neighbors *k* in kNN.
Here optimal *h* is found using LOO (epanechnikov kernel, sepal.width & sepal.length only): 

![LOOker1parzen2](https://github.com/toxazol/machineLearning/blob/master/img/LOOker1parzen2.png?raw=true)

all four features:

![LOOker1parzen4](https://github.com/toxazol/machineLearning/blob/master/img/LOOker1parzen4.png?raw=true)

triangle kernel, h=0.4:

![h04ker3parzen2](https://github.com/toxazol/machineLearning/blob/master/img/h04ker3parzen2.png?raw=true)

gaussian kernel, h=0.1:

![h01ker4parzen2](https://github.com/toxazol/machineLearning/blob/master/img/h01ker4parzen2.png?raw=true)

Parzen window algorithm can be modified to suit case-based reasoning better.
It's what we call **parzen window algorithm with variable window width**.
Let *h* be equal to the distance to *k+1* nearest neighbor.
Here is comparison of parzen window classifier (uniform kernel) without and with variable window width modification applied:
![parzenKer5](https://github.com/toxazol/machineLearning/blob/master/img/parzenKer5.png?raw=true)
![parzenKer5Var](https://github.com/toxazol/machineLearning/blob/master/img/parzenKer5Var.png?raw=true)








 






