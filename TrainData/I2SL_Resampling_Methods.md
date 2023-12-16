# Resampling Methods

In statistics, resampling is any of a variety of methods for doing one of the following:

* Estimating the precision of sample statistics (medians, variances, percentiles) by using subsets of available data (jackknifing) or drawing randomly with replacement from a set of data points (bootstrapping)  
* Exchanging labels on data points when performing significance tests (permutation tests, also called exact tests, randomization tests, or re-randomization tests)  
* Validating models by using random subsets (bootstrapping, cross-validation)  

## Bootstrap


Bootstrapping is a statistical method for estimating the sampling distribution] of an estimator by sampling with replacement from the original sample, most often with the purpose of deriving robust estimates of standard errors and confidence intervals of a
population parameter like a mean, median, proportion, odds ratio, correlation coefficient or regression coefficient. It has been called the **plug-in principle**, as it is the method of estimation of functionals of a population distribution by evaluating
the same functionals at the empirical distribution based on a sample. It is called a principl because it is too simple to be otherwise, it is just a guideline, not a theorem.   

![Bootstrapping](https://raw.githubusercontent.com/nikbearbrown/Google_Colab/master/img/Bootstrapping.jpg)
_plug-in principle of Bootstrapping_


It may also be used for constructing hypothesis tests. It is often used as a robust alternative to inference based on parametric assumptions when those assumptions are in doubt, or where parametric inference is impossible or requires very complicated formulas for the calculation of standard errors. Bootstrapping techniques are also used in the updating-selection transitions of particle filters, genetic type
algorithms and related resample/reconfiguration Monte Carlo methods used in computational physics.  

Todo:  Psuedocode for bootstrap and and example in python


### Cross Validation  

Usually a test set is not available so a simple strategy to create one is to split the available data into training and testing (validation set). For quantitative responses usually use MSE, for categorical can use error rate, area under the curve, F1 score, weighting of confusion matrix, etc...

## Jackknife resampling

In statistics, the jackknife is a resampling technique especially useful for variance and bias estimation. The jackknife pre-dates other common resampling methods such as the bootstrap. The jackknife estimator of a parameter is found by systematically leaving out each observation from a dataset and calculating the estimate and then finding the average of these calculations. Given a sample of size $n$, the jackknife estimate is found by aggregating the estimates of each $(n-1)$-sized sub-sample.

## Leave One Out Cross Validation

LOOCV has only one observation in the test set and uses all other n-1 observations to build a model. n different models are built leaving out each observation once and error is averaged over these n trials.  LOOCV is better than simple method above. Model is built on nearly all the data and there is no randomness in the splits since each observation will be left out once. It is computationally expensive especially with large n and a complex model.

## Cross-validation  

Cross-validation is a statistical method for validating a predictive model. Subsets of the data are held out for use as validating sets; a model is fit to the remaining data (a training set) and used to predict
for the validation set. Averaging the quality of the predictions across the validation sets yields an overall measure of prediction accuracy. Cross-validation is employed repeatedly in building decision trees.

One form of cross-validation leaves out a single observation at a time; this is similar to the _jackknife_. Another, *K*-fold cross-validation,
splits the data into *K* subsets; each is held out in turn as the validation set.

This avoids \"self-influence\". For comparison, in regression analysis methods such as linear regression, each *y* value draws the regression line toward itself, making the prediction of that value appear more accurate than it really is. Cross-validation applied to linear regression predicts the *y* value for each observation without using that observation.


## *K*-fold cross validation  

Similar to LOOCV but this time you leave some number greater than 1 out. Here, k is the number of partitions of your sample, so if you have 1000 observations and k = 10, the each fold will be 100. These 100 observations would act as your test set. Get an MSE for each fold of these 100 observations and take the average. LOOCV is a special case of k-fold CV whenever k equals the number of observations.

## bias-variance tradeoff between LOOCV and k-folds  

Since LOOCV trains on nearly all the data, the test error rate will generally be lower than k-fold and there for less biased. LOOCV will have higher variance since all n models will be very highly correlated to one another. Since the models won't differ much, the test error rate (which what CV is measuring) will vary more than k-fold which has fewer models that are less correlated with one another. A value of k between 5 and 10 is a good rule of thumb that balances the trade-off between bias and variance


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

# Questions

### 1. Using basic statistical properties of the variance, as well as singlevariable calculus, derive

![image.png](data:image/png;base64,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)

### In other words, prove that α given by the above equation does indeed minimize $$Var(\alpha X + (1 - \alpha)Y)$$


### Solution:
Properties of variance and covariance
$$=Var(\alpha X) + Var((1 - \alpha)Y) + 2Cov(\alpha X, (1 - \alpha)Y)$$
$$=\alpha^2Var(X) + (1 - \alpha)^2Var(Y) + 2(\alpha)(1 - \alpha)(Cov(X, Y)$$

Take derivative and set to 0
$$2\alpha Var(X) - 2(1 - \alpha)Var(Y) + (2 - 4\alpha)Cov(X, Y) = 0$$
Collect terms
$$2\alpha Var(X) + 2 \alpha Var(Y) - 4\alpha Cov(X, Y) = 2Var(Y) - 2Cov(X, Y)$$
Solve for $\alpha$
$$\alpha = \frac{Var(Y) - Cov(X, Y)}{Var(X) + Var(Y) - 2Cov(X, Y)}$$

### 2. We will now derive the probability that a given observation is part of a bootstrap sample. Suppose that we obtain a bootstrap sample from a set of n observations.
(a) What is the probability that the first bootstrap observation is
not the jth observation from the original sample? Justify your
answer.

(b) What is the probability that the second bootstrap observation
is not the jth observation from the original sample?

(c) Argue that the probability that the jth observation is not in the
bootstrap sample is (1 − 1/n)n.

(d) When n = 5, what is the probability that the jth observation is
in the bootstrap sample?

(e) When n = 100, what is the probability that the jth observation
is in the bootstrap sample?


### Soutions:
a) $\frac{n-1}{n}$  
b) $\frac{n-1}{n}$  
c) Since bootstrapping is sampling with replace, the probability of being any jth obsevation is $\frac{1}{n}$. The probability of not being the jth observation is $1 - \frac{1}{n}$. Since each draw is independent we can just multiply the probabilities together to get the probability that the jth observation is not in the sample at all


```python
#2 c-f
[(1 - 1/n) **n for n in [5, 100, 10000, 100000]]
```




    [0.3276800000000001,
     0.3660323412732292,
     0.36786104643297046,
     0.3678776017682465]




```python
x = np.arange(1, 100001)
y = (1 - 1/x) ** x
```


```python
plt.plot(x[:10], y[:10])
```




    [<matplotlib.lines.Line2D at 0x7fcea4a1d710>]




    
![png](output_9_1.png)
    



```python
#2h
# make 10,000 samples of 100 elements each sample from integers 1 - 100
# check if 4 is each sample. Take mean.
# Looks like very close to theoretical probability
data = np.random.randint(1, 101, (100, 10000))
np.any(data == 4, axis=0).mean()
```




    0.6274



###3. We now review k-fold cross-validation.

(a) Explain how k-fold cross-validation is implemented.

(b) What are the advantages and disadvantages of k-fold crossvalidation relative to:

i. The validation set approach?

ii. LOOCV?

### Solutions:
a) K-fold CV works by taking the dataset given and randomly splitting it into k non-overlapping datasets. You can shuffle the data first and then just split at regular intervals. Train K models. For each model, use the kth region as the validation set and build on the other k-1 sets. Take the mean of the k errors found to estimate the true test error.  

b i) Advantage to validation set is that there are more test sets to validate on which should reduce the bias of what the overall error actually is. Variance should also decrease as the validation set approach is just one split of the data and that split could not represent the test data well. Disadvantage is training more models.  

b ii) Advantage to LOOCV is a decrease in variance as the k models are not as highly correlated as the each LOOCV model is. Also, K-folds is computationally less expensive.

### 4. Suppose that we use some statistical learning method to make a prediction for the response Y for a particular value of the predictor X. Carefully describe how we might estimate the standard deviation of our prediction.

### Answer:
* Using the bootstrap, create many (say 10,000) samples of your data.
* Create each sample by drawing n times (where n is number of observations in your original) with replacement.
* Build your model for each sample and calculate the mean and standard deviation of estimated parameters

### 5. Earlier, we used logistic regression to predict the probability of default using income and balance on the Default data set. We will now estimate the test error of this logistic regression model using the validation set approach. Do not forget to set a random seed before beginning your analysis.



### (a) Fit a logistic regression model that uses income and balance to predict default.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
```


```python
default = pd.read_csv('https://raw.githubusercontent.com/nikbearbrown/Google_Colab/master/data/default.csv')
default['student_yes'] = (default['student'] == 'Yes').astype('int')
default['default_yes'] = (default['default'] == 'Yes').astype('int')
```


```python
default.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>default</th>
      <th>student</th>
      <th>balance</th>
      <th>income</th>
      <th>student_yes</th>
      <th>default_yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>No</td>
      <td>No</td>
      <td>729.526495</td>
      <td>44361.625074</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>No</td>
      <td>Yes</td>
      <td>817.180407</td>
      <td>12106.134700</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>No</td>
      <td>No</td>
      <td>1073.549164</td>
      <td>31767.138947</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>No</td>
      <td>No</td>
      <td>529.250605</td>
      <td>35704.493935</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>No</td>
      <td>No</td>
      <td>785.655883</td>
      <td>38463.495879</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = default[['balance', 'income']]
y = default['default_yes']
```

# No Validation set

### Sklearn


```python
# Notice how tol must be changed to less than default value or convergence won't happen
# Use a high value of C to remove regularization
model = LogisticRegression(C=100000, tol=.0000001)
model.fit(X, y)
model.intercept_, model.coef_
```




    (array([-11.54046839]), array([[5.64710291e-03, 2.08089921e-05]]))



### Statsmodels
Coefficients are similar


```python
import statsmodels.formula.api as smf
```

    /usr/local/lib/python3.6/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm
    


```python
result = smf.logit(formula='default_yes ~ balance + income', data=default).fit()
```

    Optimization terminated successfully.
             Current function value: 0.078948
             Iterations 10
    


```python
result.summary()
```




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>default_yes</td>   <th>  No. Observations:  </th>   <td> 10000</td>  
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>   <td>  9997</td>  
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>   <td>     2</td>  
</tr>
<tr>
  <th>Date:</th>            <td>Mon, 21 Sep 2020</td> <th>  Pseudo R-squ.:     </th>   <td>0.4594</td>  
</tr>
<tr>
  <th>Time:</th>                <td>06:03:04</td>     <th>  Log-Likelihood:    </th>  <td> -789.48</td> 
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th>  <td> -1460.3</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th> <td>4.541e-292</td>
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>  -11.5405</td> <td>    0.435</td> <td>  -26.544</td> <td> 0.000</td> <td>  -12.393</td> <td>  -10.688</td>
</tr>
<tr>
  <th>balance</th>   <td>    0.0056</td> <td>    0.000</td> <td>   24.835</td> <td> 0.000</td> <td>    0.005</td> <td>    0.006</td>
</tr>
<tr>
  <th>income</th>    <td> 2.081e-05</td> <td> 4.99e-06</td> <td>    4.174</td> <td> 0.000</td> <td>  1.1e-05</td> <td> 3.06e-05</td>
</tr>
</table><br/><br/>Possibly complete quasi-separation: A fraction 0.14 of observations can be<br/>perfectly predicted. This might indicate that there is complete<br/>quasi-separation. In this case some parameters will not be identified.



### Error without validation set
This is an in-sample prediction. Training error in both sklearn and statsmodels. Both are equivalent


```python
(model.predict(X) == y).mean()
```




    0.9737




```python
((result.predict(X) > .5) * 1 == y).mean()
```




    0.9737



###(b) Using the validation set approach, estimate the test error of this model. In order to do this, you must perform the following steps:
i. Split the sample set into a training set and a validation set.

ii. Fit a multiple logistic regression model using only the training
observations.

iii. Obtain a prediction of default status for each individual in
the validation set by computing the posterior probability of
default for that individual, and classifying the individual to
the default category if the posterior probability is greater
than 0.5.

iv. Compute the validation set error, which is the fraction of
the observations in the validation set that are misclassified.


```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
```


```python
model = LogisticRegression(C=100000, tol=.0000001)
model.fit(X_train, y_train)
model.intercept_, model.coef_
```




    (array([-1.56156741e-06]), array([[ 0.00033632, -0.00012504]]))




```python
X_train_sm = X_train.join(y_train)
```


```python
result = smf.logit(formula='default_yes ~ balance + income', data=X_train_sm).fit()
result.summary()
```

    Optimization terminated successfully.
             Current function value: 0.078708
             Iterations 10
    




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>default_yes</td>   <th>  No. Observations:  </th>   <td>  7500</td>  
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>   <td>  7497</td>  
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>   <td>     2</td>  
</tr>
<tr>
  <th>Date:</th>            <td>Mon, 21 Sep 2020</td> <th>  Pseudo R-squ.:     </th>   <td>0.4352</td>  
</tr>
<tr>
  <th>Time:</th>                <td>06:03:04</td>     <th>  Log-Likelihood:    </th>  <td> -590.31</td> 
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th>  <td> -1045.1</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th> <td>3.068e-198</td>
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>  -11.2449</td> <td>    0.495</td> <td>  -22.704</td> <td> 0.000</td> <td>  -12.216</td> <td>  -10.274</td>
</tr>
<tr>
  <th>balance</th>   <td>    0.0054</td> <td>    0.000</td> <td>   21.084</td> <td> 0.000</td> <td>    0.005</td> <td>    0.006</td>
</tr>
<tr>
  <th>income</th>    <td> 2.132e-05</td> <td> 5.84e-06</td> <td>    3.653</td> <td> 0.000</td> <td> 9.88e-06</td> <td> 3.28e-05</td>
</tr>
</table><br/><br/>Possibly complete quasi-separation: A fraction 0.12 of observations can be<br/>perfectly predicted. This might indicate that there is complete<br/>quasi-separation. In this case some parameters will not be identified.




```python
# Nearly the same as training set. So not too much over fitting has happened
(model.predict(X_test) == y_test).mean(), ((result.predict(X_test) > .5) * 1 == y_test).mean()
```




    (0.9604, 0.9744)



Validation error of only .0272

### (c) Repeat the process in (b) three times, using three different splits of the observations into a training set and a validation set. Comment on the results obtained.


```python
# c) repeat for 3 different validation sets
model = LogisticRegression(C=100000, tol=.0000001)

for i in range(3):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)

    X_train_sm = X_train.join(y_train)
    result = smf.logit(formula='default_yes ~ balance + income', data=X_train_sm).fit()
    print((model.predict(X_test) == y_test).mean(), ((result.predict(X_test) > .5) * 1 == y_test).mean())
```

    Optimization terminated successfully.
             Current function value: 0.081600
             Iterations 10
    0.9684 0.9752
    Optimization terminated successfully.
             Current function value: 0.079349
             Iterations 10
    0.9756 0.9756
    Optimization terminated successfully.
             Current function value: 0.078415
             Iterations 10
    0.9632 0.9704
    

### (d) Now consider a logistic regression model that predicts the probability of default using income, balance, and a dummy variablefor student. Estimate the test error for this model using the validation set approach. Comment on whether or not including a dummy variable for student leads to a reduction in the test error rate.


```python
# d) include student in model
X = default[['balance', 'income', 'student_yes']]
y = default['default_yes']

model = LogisticRegression(C=100000, tol=.0000001)

for i in range(3):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)

    X_train_sm = X_train.join(y_train)
    result = smf.logit(formula='default_yes ~ balance + income + student_yes', data=X_train_sm).fit()
    print((model.predict(X_test) == y_test).mean(), ((result.predict(X_test) > .5) * 1 == y_test).mean())
```

    Optimization terminated successfully.
             Current function value: 0.081802
             Iterations 10
    0.9708 0.9784
    Optimization terminated successfully.
             Current function value: 0.082798
             Iterations 10
    0.9704 0.9776
    Optimization terminated successfully.
             Current function value: 0.078762
             Iterations 10
    0.9664 0.97
    

Looks like error rate is very similar

### 6. Computing stand errors of coefficents of logistic regression using bootstrap

We continue to consider the use of a logistic regression model to
predict the probability of default using income and balance on the
Default data set. In particular, we will now compute estimates for
the standard errors of the income and balance logistic regression coefficients
in two different ways: (1) using the bootstrap, and (2) using
the standard formula for computing the standard errors in the glm()
function. Do not forget to set a random seed before beginning your
analysis.

(a) Using the summary() and glm() functions, determine the estimated
standard errors for the coefficients associated with income
and balance in a multiple logistic regression model that uses
both predictors.

(b) Write a function, boot.fn(), that takes as input the Default data
set as well as an index of the observations, and that outputs
the coefficient estimates for income and balance in the multiple
logistic regression model.

(c) Use the boot() function together with your boot.fn() function to
estimate the standard errors of the logistic regression coefficients
for income and balance.

(d) Comment on the estimated standard errors obtained using the
glm() function and using your bootstrap function.


```python
result = smf.logit(formula='default_yes ~ balance + income', data=default).fit()
result.summary()
```

    Optimization terminated successfully.
             Current function value: 0.078948
             Iterations 10
    




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>default_yes</td>   <th>  No. Observations:  </th>   <td> 10000</td>  
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>   <td>  9997</td>  
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>   <td>     2</td>  
</tr>
<tr>
  <th>Date:</th>            <td>Mon, 21 Sep 2020</td> <th>  Pseudo R-squ.:     </th>   <td>0.4594</td>  
</tr>
<tr>
  <th>Time:</th>                <td>06:03:05</td>     <th>  Log-Likelihood:    </th>  <td> -789.48</td> 
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th>  <td> -1460.3</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th> <td>4.541e-292</td>
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>  -11.5405</td> <td>    0.435</td> <td>  -26.544</td> <td> 0.000</td> <td>  -12.393</td> <td>  -10.688</td>
</tr>
<tr>
  <th>balance</th>   <td>    0.0056</td> <td>    0.000</td> <td>   24.835</td> <td> 0.000</td> <td>    0.005</td> <td>    0.006</td>
</tr>
<tr>
  <th>income</th>    <td> 2.081e-05</td> <td> 4.99e-06</td> <td>    4.174</td> <td> 0.000</td> <td>  1.1e-05</td> <td> 3.06e-05</td>
</tr>
</table><br/><br/>Possibly complete quasi-separation: A fraction 0.14 of observations can be<br/>perfectly predicted. This might indicate that there is complete<br/>quasi-separation. In this case some parameters will not be identified.




```python
df_params = pd.DataFrame(columns=['Intercept', 'balance', 'income'])
for i in range(100):
    default_sample = default.sample(len(default), replace=True)
    result_sample = smf.logit(formula='default_yes ~ balance + income', data=default_sample).fit(disp=0)
    df_params = df_params.append(result_sample.params, ignore_index=True)
```


```python
# bootstrap parameters and standard error
df_params.mean(), df_params.std()
```




    (Intercept   -11.620645
     balance       0.005682
     income        0.000021
     dtype: float64, Intercept    0.374106
     balance      0.000209
     income       0.000005
     dtype: float64)




```python
# model parameters and standard error
result.params, result.bse
```




    (Intercept   -11.540468
     balance       0.005647
     income        0.000021
     dtype: float64, Intercept    0.434772
     balance      0.000227
     income       0.000005
     dtype: float64)



Standard errors are a wee bit higher in bootstrap

# 7
a) Fit Logistic Regression with Lag1, Lag2


```python
weekly = pd.read_csv('https://raw.githubusercontent.com/nikbearbrown/Google_Colab/master/data/weekly.csv')
```


```python
weekly['Direction_Up'] = (weekly['Direction'] == 'Up').astype(int)
```


```python
weekly.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Lag1</th>
      <th>Lag2</th>
      <th>Lag3</th>
      <th>Lag4</th>
      <th>Lag5</th>
      <th>Volume</th>
      <th>Today</th>
      <th>Direction</th>
      <th>Direction_Up</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1990</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>-3.936</td>
      <td>-0.229</td>
      <td>-3.484</td>
      <td>0.154976</td>
      <td>-0.270</td>
      <td>Down</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1990</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>-3.936</td>
      <td>-0.229</td>
      <td>0.148574</td>
      <td>-2.576</td>
      <td>Down</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1990</td>
      <td>-2.576</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>-3.936</td>
      <td>0.159837</td>
      <td>3.514</td>
      <td>Up</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1990</td>
      <td>3.514</td>
      <td>-2.576</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>0.161630</td>
      <td>0.712</td>
      <td>Up</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1990</td>
      <td>0.712</td>
      <td>3.514</td>
      <td>-2.576</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>0.153728</td>
      <td>1.178</td>
      <td>Up</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = weekly[['Lag1', 'Lag2']]
y = weekly['Direction_Up']
```


```python
model = LogisticRegression(C=100000, tol=.0000001)
model.fit(X, y)
```




    LogisticRegression(C=100000, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=1e-07, verbose=0,
                       warm_start=False)




```python
model.intercept_, model.coef_
```




    (array([0.22122405]), array([[-0.03872222,  0.0602483 ]]))




```python
# accuracy
(model.predict(X) == y).mean()
```




    0.5555555555555556



### b) Fit without first observation


```python
# model is different but nearly identical
model.fit(X.iloc[1:], y.iloc[1:])
model.intercept_, model.coef_, (model.predict(X) == y).mean()
```




    (array([0.22324305]), array([[-0.03843317,  0.06084763]]), 0.5564738292011019)




```python
# c
# wrong prediction
model.predict([X.iloc[0]]), y[0]
```




    (array([1]), 0)




```python
# d
errors = np.zeros(len(X))
for i in range(len(X)):
    leave_out  = ~X.index.isin([i])
    model.fit(X[leave_out], y[leave_out])
    if model.predict([X.iloc[i]]) != y[i]:
        errors[i] = 1
```


```python
# e
errors.mean()
```




    0.44995408631772266



## 8. We will now perform cross-validation on a simulated data set.
###(a) Generate a simulated data set.


```python
np.random.seed(1)
x = np.random.randn(100)
e = np.random.randn(100)
y = x - 2*x**2 + e
```


```python
y.shape
```




    (100,)



### (b) Create a scatterplot of X against Y . Comment on what you find.


```python
plt.scatter(x, y);
```


    
![png](output_65_0.png)
    


### (c) Set a random seed, and then compute the LOOCV errors that result from fitting the following four models using least squares:
i. Y = β0 + β1X + ǫ

ii. Y = β0 + β1X + β2X2 + ǫ

iii. Y = β0 + β1X + β2X2 + β3X3 + ǫ

iv. Y = β0 + β1X + β2X2 + β3X3 + β4X4 + ǫ.

Note you may find it helpful to use the data.frame() function
to create a single data set containing both X and Y .


```python
df = pd.DataFrame(np.array([np.ones(len(x)), x, x ** 2, x ** 3, x ** 4, y]).T, columns=['b0', 'x', 'x2', 'x3', 'x4', 'y'])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b0</th>
      <th>x</th>
      <th>x2</th>
      <th>x3</th>
      <th>x4</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>-0.416758</td>
      <td>0.173687</td>
      <td>-0.072385</td>
      <td>0.030167</td>
      <td>0.397389</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>-0.056267</td>
      <td>0.003166</td>
      <td>-0.000178</td>
      <td>0.000010</td>
      <td>0.323479</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-2.136196</td>
      <td>4.563334</td>
      <td>-9.748176</td>
      <td>20.824015</td>
      <td>-12.395997</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.640271</td>
      <td>2.690488</td>
      <td>4.413129</td>
      <td>7.238727</td>
      <td>-3.307613</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>-1.793436</td>
      <td>3.216411</td>
      <td>-5.768426</td>
      <td>10.345301</td>
      <td>-8.530344</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.linear_model import LinearRegression
```


```python
X = df.iloc[:, :5]
y = df['y']
model = LinearRegression()
errors = np.zeros((len(X), 4))
for i in range(len(X)):
    leave_out  = ~X.index.isin([i])
    for j in range(4):
        model.fit(X.iloc[leave_out, :j+2], y[leave_out])
        errors[i, j] = (model.predict([X.iloc[i, :j+2]]) - y[i]) ** 2
```


```python
# each error here is average error for linear, quadratic, cubic and quartic model.
# Looks like it stabilizes at quadratic.
errors.mean(axis=0)
```




    array([6.26076433, 0.91428971, 0.92687688, 0.86691169])




```python
# again with different seed.
np.random.seed(2)
x = np.random.randn(100)
e = np.random.randn(100)
y = x - 2*x**2 + e
df = pd.DataFrame(np.array([np.ones(len(x)), x, x ** 2, x ** 3, x ** 4, y]).T, columns=['b0', 'x', 'x2', 'x3', 'x4', 'y'])


X = df.iloc[:, :5]
y = df['y']
model = LinearRegression()
errors = np.zeros((len(X), 4))
for i in range(len(X)):
    leave_out  = ~X.index.isin([i])
    for j in range(4):
        model.fit(X.iloc[leave_out, :j+2], y[leave_out])
        errors[i, j] = (model.predict([X.iloc[i, :j+2]]) - y[i]) ** 2

# quite a different average error. But again stabilizes at quadratic which makes sense
errors.mean(axis=0)
```




    array([11.61020827,  1.26528394,  1.28204182,  1.31659158])



### f. Comment on the statistical significance of the coefficient estimates that results from fitting each of the models in (c) using least squares. Do these results agree with the conclusions drawn based on the cross-validation results?

### Answer:
since the error doesn't improve after quadratic it's likely the
standard errors for x3 and x4 would not be significant

## 9. We will now consider the Boston housing data set, from the MASS library.


```python
boston = pd.read_csv('https://raw.githubusercontent.com/nikbearbrown/Google_Colab/master/data/boston.csv')
boston.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>crim</th>
      <th>zn</th>
      <th>indus</th>
      <th>chas</th>
      <th>nox</th>
      <th>rm</th>
      <th>age</th>
      <th>dis</th>
      <th>rad</th>
      <th>tax</th>
      <th>ptratio</th>
      <th>black</th>
      <th>lstat</th>
      <th>medv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1</td>
      <td>296</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>



### (a) Based on this data set, provide an estimate for the population mean of medv. Call this estimate ˆμ.


```python
#a
boston['medv'].mean()
```




    22.532806324110698



### (b) Provide an estimate of the standard error of ˆμ. Interpret this result.
Hint: We can compute the standard error of the sample mean by
dividing the sample standard deviation by the square root of the
number of observations.


```python
#b
# standard deviation of mean
boston['medv'].std() / np.sqrt(len(boston))
```




    0.4088611474975351



### (c) Now estimate the standard error of ˆμ using the bootstrap. How does this compare to your answer from (b)?


```python
#c
#bootstrap standard deviation of mean
means = [boston['medv'].sample(n = len(boston), replace=True).mean() for i in range(1000)]
np.std(means)
```




    0.39792811024263747



### (d) Based on your bootstrap estimate from (c), provide a 95% confidence interval for the mean of medv. Compare it to the results obtained using t.test(Boston$medv).
Hint: You can approximate a 95% confidence interval using the
formula [ˆμ − 2SE(ˆμ), ˆμ + 2SE(ˆμ)].


```python
#d
se = np.std(means)
boston['medv'].mean() - 2 * se, boston['medv'].mean() + 2 * se
```




    (21.736950103625425, 23.32866254459597)



http://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data


```python
import scipy.stats as st
```


```python
st.t.interval(0.95, len(boston['medv'])-1, loc=np.mean(boston['medv']), scale=st.sem(boston['medv']))
```




    (21.729528014578616, 23.33608463364278)



### (e) Based on this data set, provide an estimate, ˆμmed, for the median value of medv in the population.


```python
#e
boston['medv'].median()
```




    21.2



### (f) We now would like to estimate the standard error of ˆμmed. Unfortunately, there is no simple formula for computing the standar error of the median. Instead, estimate the standard error of the median using the bootstrap. Comment on your findings.


```python
#f
medians = [boston['medv'].sample(n = len(boston), replace=True).median() for i in range(1000)]
np.std(medians)
```




    0.36780536972697897



### (g) Based on this data set, provide an estimate for the tenth percentile of medv in Boston suburbs. Call this quantity ˆμ0.1. (You can use the quantile() function.)


```python
#g
boston['medv'].quantile(.1)
```




    12.75



### (h) Use the bootstrap to estimate the standard error of ˆμ0.1. Comment on your findings.


```python
#h
quantile_10 = [boston['medv'].sample(n = len(boston), replace=True).quantile(.1) for i in range(1000)]
np.std(quantile_10)
```




    0.4996360575458901



### End of Chapter 5
