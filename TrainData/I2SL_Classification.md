# Classification

In statistics, classification is the problem of identifying to which of a set of categories (sub-populations) a new observation belongs, on the basis of a training set of data containing observations (or instances) whose category membership is known. Examples are assigning a given email to the "spam" or "non-spam" class, and assigning a diagnosis to a given patient based on observed characteristics of the patient (sex, blood pressure, presence or absence of certain symptoms, etc.). Classification is an example of pattern recognition.

We will discuss qualitative (categorical) variables (classification problems). Classification problems can be thought of as regression problems since most of the models return a probability of being in a certain class.


Since probabilities take on real values the problem can take the form regression with a threshold to decide the classes - as in logistic regression.  


```python
import pandas as pd
import pandas.util.testing as tm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
%matplotlib inline
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:2: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      
    


```python
default = pd.read_csv("https://raw.githubusercontent.com/nikbearbrown/Google_Colab/master/data//default.csv")
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>No</td>
      <td>No</td>
      <td>729.526495</td>
      <td>44361.625074</td>
    </tr>
    <tr>
      <th>1</th>
      <td>No</td>
      <td>Yes</td>
      <td>817.180407</td>
      <td>12106.134700</td>
    </tr>
    <tr>
      <th>2</th>
      <td>No</td>
      <td>No</td>
      <td>1073.549164</td>
      <td>31767.138947</td>
    </tr>
    <tr>
      <th>3</th>
      <td>No</td>
      <td>No</td>
      <td>529.250605</td>
      <td>35704.493935</td>
    </tr>
    <tr>
      <th>4</th>
      <td>No</td>
      <td>No</td>
      <td>785.655883</td>
      <td>38463.495879</td>
    </tr>
  </tbody>
</table>
</div>




```python
# make column for Yes defaults
default['yes'] = (default['default'] == 'Yes').astype(int)
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
      <th>yes</th>
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
    </tr>
    <tr>
      <th>1</th>
      <td>No</td>
      <td>Yes</td>
      <td>817.180407</td>
      <td>12106.134700</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>No</td>
      <td>No</td>
      <td>1073.549164</td>
      <td>31767.138947</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>No</td>
      <td>No</td>
      <td>529.250605</td>
      <td>35704.493935</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>No</td>
      <td>No</td>
      <td>785.655883</td>
      <td>38463.495879</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.lmplot('balance', 'income', data=default.sample(1000), hue='default', fit_reg=False)
plt.ylim([0,70000])
plt.xlim([-100,2500])
```




    (-100.0, 2500.0)




    
![png](output_5_1.png)
    



```python
sns.boxplot('student', 'balance', data=default, hue='default');
```


    
![png](output_6_0.png)
    



```python
sns.boxplot('yes', 'balance', data=default, hue='default');
```


    
![png](output_7_0.png)
    


## Why not linear regression

Though it is always possible to use numeric values for the categories of the response, there generally is no natural way to order and separate the values in a way that makes sense. Only in a two-category problem will the ordering make sense but even then linear regression will produce probability estimates outside of [0, 1].


```python
# Looks quite a bit different than the linear regression model
sns.lmplot('balance', 'yes', data=default, logistic=True)
```




    <seaborn.axisgrid.FacetGrid at 0x7fc9bf921a20>




    
![png](output_9_1.png)
    


## Logistic regression

Find function that always outputs number between 0 and 1. Many functions satisfy this condition. For logistic regression the ... logistic function! is used.

$$y = \frac{e^{\beta_0 + \beta_1X}}{1 + e^{\beta_0 + \beta_1X}}$$

Many times you will see this as the sigmoid function in a simpler format

$$y = \frac{1}{1 + e^{-t}}$$

Where $t$ is just the normal linear model $t = \beta_0 + \beta_1X$. Some algebra can be used to show the two equations above are equivalent.

y can now be thought as the probability given some value X since it will always be between 0 and 1. Some more algebra can show that $$log{\frac{p(X)}{1 - p(X)}} = \beta_0 + \beta_1X$$

Where $y$ has been replaced by $p(X)$, the probability of $X$. The expression $\frac{p(X)}{1 - p(X)}$ is known as the 'odds'. So for instance if you are gambling and think that Clinton will win the presidency 80% of the time. The odds would be .8/.2 = 4 or said "4 to 1". For every 4 times she wins, Trump will win once.

What logistic regression is saying, that the log odds are modeled by a linear model which can be solved by linear regression. This has the literal meaning of - given a one unit increase in one of the variables (say $X_1$), a $\beta_1$ increase will occur to the log-odds. Or equivalently, the odds will be multiplied by $e^{\beta_1}$.

In our election example, $X_1$ could be the percentage of voters under 30 and $\beta_1$ could be .5. That would mean if $X_1$ were to increase by 1 percentage point, Clinton's log odds would increase by .5. In our example from above, Clinton's log odds would go from 4 to 4.5 and her probability of winning would go from 80% to 4.5 / 5.5 or 82%

There is no straight-line relationship between the probability of being in a certain class and X in logistic regression because of the need to have probabilities between 0 and 1.

## Estimating coefficients through Maximum Likelihood
In linear regression, the model coefficients were found by minimizing the squared residuals. In logistic regression, we maximize the probabilities of all points by a method called maximum likelihood. Maximum likelihood multiplies the model probability for each observation together and chooses parameters that maximize this number. The log likelihood is actually used as numerical underflow will be a problem for must problems with a non-trivial amount of data.  



```python
results = smf.logit('yes ~ balance', data=default).fit()
```

    Optimization terminated successfully.
             Current function value: 0.079823
             Iterations 10
    


```python
results.summary()
```




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>yes</td>       <th>  No. Observations:  </th>   <td> 10000</td>  
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>   <td>  9998</td>  
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>   <td>     1</td>  
</tr>
<tr>
  <th>Date:</th>            <td>Wed, 23 Sep 2020</td> <th>  Pseudo R-squ.:     </th>   <td>0.4534</td>  
</tr>
<tr>
  <th>Time:</th>                <td>07:35:12</td>     <th>  Log-Likelihood:    </th>  <td> -798.23</td> 
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th>  <td> -1460.3</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th> <td>6.233e-290</td>
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>  -10.6513</td> <td>    0.361</td> <td>  -29.491</td> <td> 0.000</td> <td>  -11.359</td> <td>   -9.943</td>
</tr>
<tr>
  <th>balance</th>   <td>    0.0055</td> <td>    0.000</td> <td>   24.952</td> <td> 0.000</td> <td>    0.005</td> <td>    0.006</td>
</tr>
</table><br/><br/>Possibly complete quasi-separation: A fraction 0.13 of observations can be<br/>perfectly predicted. This might indicate that there is complete<br/>quasi-separation. In this case some parameters will not be identified.



## Interpretation

For every one dollar increase in balance the log odds increases by 0.00555. The log odds when there is no balance is -10.6.  

## Logistic Regression with regularization

The sklearn Logistic Regression _LogisticRegression()_ uses regularization by default.  See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

The regularization strength can also be adjusted by the hyperparamtere _C_.

_C_ float, default=1.0

Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.



```python
# this actually uses regularization by default
lr = LogisticRegression()
```


```python
X = np.column_stack((np.ones(len(default)), default['balance']))
```


```python
lr.fit(X, default['yes'])
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)




```python
# Model coefficients are different by B1 is very similar
lr.coef_
```




    array([[-5.16481986,  0.00530575]])




```python
# predict 2000 dollar balance default
lr.predict_proba([[1, 1000]]) # 99 percent chance no default
```




    array([[0.99346413, 0.00653587]])




```python
# predict 2000 dollar balance default
lr.predict_proba([[1, 2000]]) ## 55 percent chance default
```




    array([[0.42999758, 0.57000242]])




```python
# predict 3000 dollar balance default
lr.predict_proba([[1, 3000]]) ## >99 percent chance default
```




    array([[0.00372998, 0.99627002]])



##  Multiple Linear Regression

Multiple predictors.


```python
results = smf.logit('yes ~ balance + student', data=default).fit()
```

    Optimization terminated successfully.
             Current function value: 0.078584
             Iterations 10
    


```python
results.summary()
```




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>yes</td>       <th>  No. Observations:  </th>   <td> 10000</td>  
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>   <td>  9997</td>  
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>   <td>     2</td>  
</tr>
<tr>
  <th>Date:</th>            <td>Wed, 23 Sep 2020</td> <th>  Pseudo R-squ.:     </th>   <td>0.4619</td>  
</tr>
<tr>
  <th>Time:</th>                <td>07:35:12</td>     <th>  Log-Likelihood:    </th>  <td> -785.84</td> 
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th>  <td> -1460.3</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th> <td>1.189e-293</td>
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>           <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>      <td>  -10.7495</td> <td>    0.369</td> <td>  -29.115</td> <td> 0.000</td> <td>  -11.473</td> <td>  -10.026</td>
</tr>
<tr>
  <th>student[T.Yes]</th> <td>   -0.7149</td> <td>    0.148</td> <td>   -4.846</td> <td> 0.000</td> <td>   -1.004</td> <td>   -0.426</td>
</tr>
<tr>
  <th>balance</th>        <td>    0.0057</td> <td>    0.000</td> <td>   24.748</td> <td> 0.000</td> <td>    0.005</td> <td>    0.006</td>
</tr>
</table><br/><br/>Possibly complete quasi-separation: A fraction 0.15 of observations can be<br/>perfectly predicted. This might indicate that there is complete<br/>quasi-separation. In this case some parameters will not be identified.



## Simpsons paradox

Simpson's paradox, which goes by several names, is a phenomenon in probability and statistics, in which a trend appears in several different groups of data but disappears or reverses when these groups are combined. This result is often encountered in social-science and medical-science statistics and is particularly problematic when frequency data is unduly given causal interpretations. The paradox can be resolved when causal relations are appropriately addressed in the statistical modeling. It is also referred to as Simpson's reversal, Yule–Simpson effect, amalgamation paradox, or reversal paradox.


![Simpson's paradox for quantitative data](https://raw.githubusercontent.com/nikbearbrown/Google_Colab/master/img/440px-Simpson's_paradox_continuous.svg.png)

_Simpson's paradox for quantitative data: a positive trend appears for two separate groups, whereas a negative trend appears when the groups are combined._


![Simpson's paradox for quantitative data](https://raw.githubusercontent.com/nikbearbrown/Google_Colab/master/img/440px-Simpsons_paradox_-_animation.gif)

_An alternative visualization of Simpson's paradox on data resembling real-world variability indicates that risk of misjudgement of true relationship can indeed be hard to spot_



```python
results = smf.logit('yes ~ student', data=default).fit()
results.summary()
```

    Optimization terminated successfully.
             Current function value: 0.145434
             Iterations 7
    




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>yes</td>       <th>  No. Observations:  </th>  <td> 10000</td>  
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td>  9998</td>  
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>     1</td>  
</tr>
<tr>
  <th>Date:</th>            <td>Wed, 23 Sep 2020</td> <th>  Pseudo R-squ.:     </th> <td>0.004097</td> 
</tr>
<tr>
  <th>Time:</th>                <td>07:35:12</td>     <th>  Log-Likelihood:    </th> <td> -1454.3</td> 
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -1460.3</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th> <td>0.0005416</td>
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>           <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>      <td>   -3.5041</td> <td>    0.071</td> <td>  -49.554</td> <td> 0.000</td> <td>   -3.643</td> <td>   -3.366</td>
</tr>
<tr>
  <th>student[T.Yes]</th> <td>    0.4049</td> <td>    0.115</td> <td>    3.520</td> <td> 0.000</td> <td>    0.179</td> <td>    0.630</td>
</tr>
</table>



The first model above with both balance and student show a negative relationship between student and default - meaning that being a student decreases the likelihood of defaulting. The second model shows the opposite, that being a student increases the chance at defaulting. This can be explained by the fact that students have more debt on average but compared to those with the same amount of debt they are less likely to default.

### Multiclass logistic regression   

The book unfortunately does not cover logistic regression with more than 2 classes, though this is a common occurrence in many real life problems.

### One vs All  

A simple method when you have k classes where k > 2 is to create k-1 independent logistic regression classifiers by choosing the response variable to be binary, 1 when in the current class else 0.  

## Linear Discriminant Analysis  

Linear discriminant analysis (LDA), normal discriminant analysis (NDA), or discriminant function analysis is a generalization of Fisher's linear discriminant, a method used in statistics, pattern recognition, and machine learning to find a linear combination of features that characterizes or separates two or more classes of objects or events. Not to be confused with latent dirichlet allocation. Used for multiclass classificaion problems.   LDA assumes all perdictor variables come from a gaussian distribution and estimates the mean and variance for each predictor variable where the variance is the same across for each predictor variable. It also estimates a prior probability simply by using the proporiton of classes in the training set. The resulting combination may be used as a linear classifier, or, more commonly, for dimensionality reduction before later classification.

LDA is closely related to analysis of variance (ANOVA) and regression analysis, which also attempt to express one dependent variable as a linear combination of other features or measurements. However, ANOVA uses categorical independent variables and a continuous dependent variable, whereas discriminant analysis has continuous independent variables and a categorical dependent variable (i.e. the class label). Logistic regression and probit regression are more similar to LDA than ANOVA is, as they also explain a categorical variable by the values of continuous independent variables.

Bayes rule is used to compute a probability for each class. When there is more than one predictor, a multivariate gaussian is used. Correlations between each predictor must be estimated (the covariance matrix) as they are a parameter to the multivariate gaussian.

### Stock market prediciton with LDA  

In these data we will try to predict whether the market will go up or down tomorrow.  




```python
smarket = pd.read_csv('https://raw.githubusercontent.com/nikbearbrown/Google_Colab/master/data/smarket.csv')
smarket.head()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001</td>
      <td>0.381</td>
      <td>-0.192</td>
      <td>-2.624</td>
      <td>-1.055</td>
      <td>5.010</td>
      <td>1.1913</td>
      <td>0.959</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>0.959</td>
      <td>0.381</td>
      <td>-0.192</td>
      <td>-2.624</td>
      <td>-1.055</td>
      <td>1.2965</td>
      <td>1.032</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001</td>
      <td>1.032</td>
      <td>0.959</td>
      <td>0.381</td>
      <td>-0.192</td>
      <td>-2.624</td>
      <td>1.4112</td>
      <td>-0.623</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>-0.623</td>
      <td>1.032</td>
      <td>0.959</td>
      <td>0.381</td>
      <td>-0.192</td>
      <td>1.2760</td>
      <td>0.614</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001</td>
      <td>0.614</td>
      <td>-0.623</td>
      <td>1.032</td>
      <td>0.959</td>
      <td>0.381</td>
      <td>1.2057</td>
      <td>0.213</td>
      <td>Up</td>
    </tr>
  </tbody>
</table>
</div>




```python
smarket['Up'] = np.where(smarket['Direction'] == 'Up', 1, 0)
smarket.head()
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
      <th>Up</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001</td>
      <td>0.381</td>
      <td>-0.192</td>
      <td>-2.624</td>
      <td>-1.055</td>
      <td>5.010</td>
      <td>1.1913</td>
      <td>0.959</td>
      <td>Up</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>0.959</td>
      <td>0.381</td>
      <td>-0.192</td>
      <td>-2.624</td>
      <td>-1.055</td>
      <td>1.2965</td>
      <td>1.032</td>
      <td>Up</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001</td>
      <td>1.032</td>
      <td>0.959</td>
      <td>0.381</td>
      <td>-0.192</td>
      <td>-2.624</td>
      <td>1.4112</td>
      <td>-0.623</td>
      <td>Down</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>-0.623</td>
      <td>1.032</td>
      <td>0.959</td>
      <td>0.381</td>
      <td>-0.192</td>
      <td>1.2760</td>
      <td>0.614</td>
      <td>Up</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001</td>
      <td>0.614</td>
      <td>-0.623</td>
      <td>1.032</td>
      <td>0.959</td>
      <td>0.381</td>
      <td>1.2057</td>
      <td>0.213</td>
      <td>Up</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
smarket.corr()
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
      <th>Up</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Year</th>
      <td>1.000000</td>
      <td>0.029700</td>
      <td>0.030596</td>
      <td>0.033195</td>
      <td>0.035689</td>
      <td>0.029788</td>
      <td>0.539006</td>
      <td>0.030095</td>
      <td>0.074608</td>
    </tr>
    <tr>
      <th>Lag1</th>
      <td>0.029700</td>
      <td>1.000000</td>
      <td>-0.026294</td>
      <td>-0.010803</td>
      <td>-0.002986</td>
      <td>-0.005675</td>
      <td>0.040910</td>
      <td>-0.026155</td>
      <td>-0.039757</td>
    </tr>
    <tr>
      <th>Lag2</th>
      <td>0.030596</td>
      <td>-0.026294</td>
      <td>1.000000</td>
      <td>-0.025897</td>
      <td>-0.010854</td>
      <td>-0.003558</td>
      <td>-0.043383</td>
      <td>-0.010250</td>
      <td>-0.024081</td>
    </tr>
    <tr>
      <th>Lag3</th>
      <td>0.033195</td>
      <td>-0.010803</td>
      <td>-0.025897</td>
      <td>1.000000</td>
      <td>-0.024051</td>
      <td>-0.018808</td>
      <td>-0.041824</td>
      <td>-0.002448</td>
      <td>0.006132</td>
    </tr>
    <tr>
      <th>Lag4</th>
      <td>0.035689</td>
      <td>-0.002986</td>
      <td>-0.010854</td>
      <td>-0.024051</td>
      <td>1.000000</td>
      <td>-0.027084</td>
      <td>-0.048414</td>
      <td>-0.006900</td>
      <td>0.004215</td>
    </tr>
    <tr>
      <th>Lag5</th>
      <td>0.029788</td>
      <td>-0.005675</td>
      <td>-0.003558</td>
      <td>-0.018808</td>
      <td>-0.027084</td>
      <td>1.000000</td>
      <td>-0.022002</td>
      <td>-0.034860</td>
      <td>0.005423</td>
    </tr>
    <tr>
      <th>Volume</th>
      <td>0.539006</td>
      <td>0.040910</td>
      <td>-0.043383</td>
      <td>-0.041824</td>
      <td>-0.048414</td>
      <td>-0.022002</td>
      <td>1.000000</td>
      <td>0.014592</td>
      <td>0.022951</td>
    </tr>
    <tr>
      <th>Today</th>
      <td>0.030095</td>
      <td>-0.026155</td>
      <td>-0.010250</td>
      <td>-0.002448</td>
      <td>-0.006900</td>
      <td>-0.034860</td>
      <td>0.014592</td>
      <td>1.000000</td>
      <td>0.730563</td>
    </tr>
    <tr>
      <th>Up</th>
      <td>0.074608</td>
      <td>-0.039757</td>
      <td>-0.024081</td>
      <td>0.006132</td>
      <td>0.004215</td>
      <td>0.005423</td>
      <td>0.022951</td>
      <td>0.730563</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = smarket[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5']].values
y = smarket['Up'].values
```


```python
train_bool = smarket['Year'].values < 2005
X_train = X[train_bool]
X_test = X[~train_bool]
y_train = y[train_bool]
y_test = y[~train_bool]
```


```python
results = smf.logit('Up ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5', data=smarket).fit()
results.summary()
```

    Optimization terminated successfully.
             Current function value: 0.691327
             Iterations 4
    




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>Up</td>        <th>  No. Observations:  </th>  <td>  1250</td> 
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td>  1244</td> 
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>     5</td> 
</tr>
<tr>
  <th>Date:</th>            <td>Wed, 23 Sep 2020</td> <th>  Pseudo R-squ.:     </th> <td>0.001651</td>
</tr>
<tr>
  <th>Time:</th>                <td>07:35:13</td>     <th>  Log-Likelihood:    </th> <td> -864.16</td>
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -865.59</td>
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th>  <td>0.7219</td> 
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    0.0742</td> <td>    0.057</td> <td>    1.309</td> <td> 0.191</td> <td>   -0.037</td> <td>    0.185</td>
</tr>
<tr>
  <th>Lag1</th>      <td>   -0.0713</td> <td>    0.050</td> <td>   -1.424</td> <td> 0.155</td> <td>   -0.170</td> <td>    0.027</td>
</tr>
<tr>
  <th>Lag2</th>      <td>   -0.0441</td> <td>    0.050</td> <td>   -0.882</td> <td> 0.378</td> <td>   -0.142</td> <td>    0.054</td>
</tr>
<tr>
  <th>Lag3</th>      <td>    0.0092</td> <td>    0.050</td> <td>    0.185</td> <td> 0.853</td> <td>   -0.089</td> <td>    0.107</td>
</tr>
<tr>
  <th>Lag4</th>      <td>    0.0072</td> <td>    0.050</td> <td>    0.145</td> <td> 0.885</td> <td>   -0.091</td> <td>    0.105</td>
</tr>
<tr>
  <th>Lag5</th>      <td>    0.0093</td> <td>    0.049</td> <td>    0.188</td> <td> 0.851</td> <td>   -0.088</td> <td>    0.106</td>
</tr>
</table>




```python
lr = LogisticRegression()
lr.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)




```python
# true on the left axis, predicted above
confusion_matrix(y_test, lr.predict(X_test))
```




    array([[ 37,  74],
           [ 30, 111]])




```python
147/ len(y_test)
```




    0.5833333333333334



Out of the 68 predicted down, 37 actually were down days.

54% accurracy  

Out of the 184 predicted up, 111 actually were up. 60% accuracy.  

58% total accuracy  


```python
y_pred = lr.predict(X_test)
```


```python
y_pred[y_test == 1]
```




    array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0,
           1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1,
           1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0,
           1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0,
           1, 1, 1, 1, 1, 1, 1, 0, 1])




```python
lda = LinearDiscriminantAnalysis()
```


```python
lda.fit(X_train, y_train)
```




    LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                               solver='svd', store_covariance=False, tol=0.0001)




```python
#almost exact same as logistic regression
confusion_matrix(y_test, lda.predict(X_test))
```




    array([[ 37,  74],
           [ 30, 111]])




```python
lda.priors_
```




    array([0.49198397, 0.50801603])




```python
# use QDA with only 2 variables
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train[:,:2], y_train)
```




    QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                                  store_covariance=False, tol=0.0001)




```python
#almost exact same as logistic regression
confusion_matrix(y_test, qda.predict(X_test[:,:2]))
```




    array([[ 30,  81],
           [ 20, 121]])




```python
knn = KNeighborsClassifier(n_neighbors=3)
```


```python
knn.fit(X_train[:,:2], y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=3, p=2,
                         weights='uniform')




```python
confusion_matrix(y_test, knn.predict(X_test[:,:2]))
```




    array([[48, 63],
           [55, 86]])



## Excercises

Question 1

Convert equation 1.1 to 1.2.  

1.1
$$p(x) = \frac{e^{\beta_0 + \beta_1X}}{1 + e^{\beta_0 + \beta_1X}}$$

1.2
$$\frac{p(x)}{1 - p(x)} = e^{\beta_0 + \beta_1X}$$

Answer 1

First, multiply 1.1 by negative 1 and add 1 to both sides
$$1 - p(x) = 1 - \frac{e^{\beta_0 + \beta_1X}}{1 + e^{\beta_0 + \beta_1X}}$$
Simplify right hand side
$$1 - p(x) = \frac{1}{1 + e^{\beta_0 + \beta_1X}}$$
Now just divide 1.2 by the last equation and you have the result




```python
# a
b0 = -6
b1 = .05
b2 = 1
x1 = 40
x2 = 3.5
t = -6 + b1 * x1 + b2 * x2
print("student has a {:.3f} probability of getting an A".format(1 / (1 + np.exp(-t))))

#b. solve for t = 0. Since an odds of 1 corresponds to 50/50 chance and log(1) = 0
# 0 = -6 + b1 * x1 + b2 * x2
hours = (6 - b2 * x2) / b1
print("student needs to study {} hours to have a 50% chance at an A".format(1 / (1 + np.exp(-t))))
```

    student has a 0.378 probability of getting an A
    student needs to study 0.3775406687981454 hours to have a 50% chance at an A
    


```python
# double check 50%
b0 = -6
b1 = .05
b2 = 1
x1 = 50
x2 = 3.5
t = -6 + b1 * x1 + b2 * x2
1 / (1 + np.exp(-t))
```




    0.5




```python
# 7
prior = .8
mu_d = 10
mu_no_d = 0
sigma = 6
normal = lambda x, m, s: 1 / np.sqrt(2 * np.pi * s ** 2) * np.exp(-(x - m) ** 2 / (2 * s ** 2))
```


```python
f_d = normal(4, 10, 6)
f_no_d = normal(4, 0, 6)
f_d, f_no_d
```




    (0.0403284540865239, 0.053241334253725375)




```python
# bayes
prob_div = prior * f_d / (prior * f_d + (1 - prior) * f_no_d)
print("Probability of dividend is {:.3f}".format(prob_div))
```

    Probability of dividend is 0.752
    


```python
# a
# p / (1 - p) = .37
# 1 / p - 1 = 1 / .37
odds = .37
one_over_p = 1 + 1 / odds
p = 1 / one_over_p
print("The probability of defaulting with odds of {} are {:.2f}".format(odds, p))
print("The odds of defaulting with probability .16 are {:.2f}".format(.16 / .84))
```

    The probability of defaulting with odds of 0.37 are 0.27
    The odds of defaulting with probability .16 are 0.19
    

# 2. The following questions should be answered using the Weekly data set. It contains 1, 089 weekly returns for 21 years, from the beginning of 1990 to the end of 2010.

(a) Produce some numerical and graphical summaries of the Weekly
data. Do there appear to be any patterns?


```python
weekly = pd.read_csv("https://raw.githubusercontent.com/nikbearbrown/Google_Colab/master/data/weekly.csv")
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
    </tr>
  </tbody>
</table>
</div>




```python
# strongest correlations with today are lag1 and lag3
weekly.corr()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Year</th>
      <td>1.000000</td>
      <td>-0.032289</td>
      <td>-0.033390</td>
      <td>-0.030006</td>
      <td>-0.031128</td>
      <td>-0.030519</td>
      <td>0.841942</td>
      <td>-0.032460</td>
    </tr>
    <tr>
      <th>Lag1</th>
      <td>-0.032289</td>
      <td>1.000000</td>
      <td>-0.074853</td>
      <td>0.058636</td>
      <td>-0.071274</td>
      <td>-0.008183</td>
      <td>-0.064951</td>
      <td>-0.075032</td>
    </tr>
    <tr>
      <th>Lag2</th>
      <td>-0.033390</td>
      <td>-0.074853</td>
      <td>1.000000</td>
      <td>-0.075721</td>
      <td>0.058382</td>
      <td>-0.072499</td>
      <td>-0.085513</td>
      <td>0.059167</td>
    </tr>
    <tr>
      <th>Lag3</th>
      <td>-0.030006</td>
      <td>0.058636</td>
      <td>-0.075721</td>
      <td>1.000000</td>
      <td>-0.075396</td>
      <td>0.060657</td>
      <td>-0.069288</td>
      <td>-0.071244</td>
    </tr>
    <tr>
      <th>Lag4</th>
      <td>-0.031128</td>
      <td>-0.071274</td>
      <td>0.058382</td>
      <td>-0.075396</td>
      <td>1.000000</td>
      <td>-0.075675</td>
      <td>-0.061075</td>
      <td>-0.007826</td>
    </tr>
    <tr>
      <th>Lag5</th>
      <td>-0.030519</td>
      <td>-0.008183</td>
      <td>-0.072499</td>
      <td>0.060657</td>
      <td>-0.075675</td>
      <td>1.000000</td>
      <td>-0.058517</td>
      <td>0.011013</td>
    </tr>
    <tr>
      <th>Volume</th>
      <td>0.841942</td>
      <td>-0.064951</td>
      <td>-0.085513</td>
      <td>-0.069288</td>
      <td>-0.061075</td>
      <td>-0.058517</td>
      <td>1.000000</td>
      <td>-0.033078</td>
    </tr>
    <tr>
      <th>Today</th>
      <td>-0.032460</td>
      <td>-0.075032</td>
      <td>0.059167</td>
      <td>-0.071244</td>
      <td>-0.007826</td>
      <td>0.011013</td>
      <td>-0.033078</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
today = weekly['Today']
today_perc = (100 + today) / 100
today_perc.cumprod().plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc9becee358>




    
![png](output_58_1.png)
    



```python
weekly['Volume'].plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc9bec6b160>




    
![png](output_59_1.png)
    



```python
plt.figure(figsize=(8, 6))
sns.boxplot('Direction', 'Lag1', data=weekly)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc9bec3c320>




    
![png](output_60_1.png)
    



```python
plt.figure(figsize=(8, 6))
sns.boxplot('Direction', 'Lag3', data=weekly)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc9bebb8240>




    
![png](output_61_1.png)
    


(b) Perform a logistic regression with
Direction as the response and the five lag variables plus Volume
as predictors. Use the summary function to print the results. Do
any of the predictors appear to be statistically significant? If so,
which ones?


```python
weekly['Direction'] = np.where(weekly['Direction'] == 'Up', 1, 0)
```


```python
results = smf.logit('Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume', data=weekly).fit()
results.summary()
```

    Optimization terminated successfully.
             Current function value: 0.682441
             Iterations 4
    




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>Direction</td>    <th>  No. Observations:  </th>  <td>  1089</td> 
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td>  1082</td> 
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>     6</td> 
</tr>
<tr>
  <th>Date:</th>            <td>Wed, 23 Sep 2020</td> <th>  Pseudo R-squ.:     </th> <td>0.006580</td>
</tr>
<tr>
  <th>Time:</th>                <td>07:35:14</td>     <th>  Log-Likelihood:    </th> <td> -743.18</td>
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -748.10</td>
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th>  <td>0.1313</td> 
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    0.2669</td> <td>    0.086</td> <td>    3.106</td> <td> 0.002</td> <td>    0.098</td> <td>    0.435</td>
</tr>
<tr>
  <th>Lag1</th>      <td>   -0.0413</td> <td>    0.026</td> <td>   -1.563</td> <td> 0.118</td> <td>   -0.093</td> <td>    0.010</td>
</tr>
<tr>
  <th>Lag2</th>      <td>    0.0584</td> <td>    0.027</td> <td>    2.175</td> <td> 0.030</td> <td>    0.006</td> <td>    0.111</td>
</tr>
<tr>
  <th>Lag3</th>      <td>   -0.0161</td> <td>    0.027</td> <td>   -0.602</td> <td> 0.547</td> <td>   -0.068</td> <td>    0.036</td>
</tr>
<tr>
  <th>Lag4</th>      <td>   -0.0278</td> <td>    0.026</td> <td>   -1.050</td> <td> 0.294</td> <td>   -0.080</td> <td>    0.024</td>
</tr>
<tr>
  <th>Lag5</th>      <td>   -0.0145</td> <td>    0.026</td> <td>   -0.549</td> <td> 0.583</td> <td>   -0.066</td> <td>    0.037</td>
</tr>
<tr>
  <th>Volume</th>    <td>   -0.0227</td> <td>    0.037</td> <td>   -0.616</td> <td> 0.538</td> <td>   -0.095</td> <td>    0.050</td>
</tr>
</table>




```python
results = smf.logit('Direction ~ Lag2', data=weekly).fit()
results.summary()
```

    Optimization terminated successfully.
             Current function value: 0.684306
             Iterations 4
    




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>Direction</td>    <th>  No. Observations:  </th>  <td>  1089</td> 
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td>  1087</td> 
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>     1</td> 
</tr>
<tr>
  <th>Date:</th>            <td>Wed, 23 Sep 2020</td> <th>  Pseudo R-squ.:     </th> <td>0.003866</td>
</tr>
<tr>
  <th>Time:</th>                <td>07:35:14</td>     <th>  Log-Likelihood:    </th> <td> -745.21</td>
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -748.10</td>
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th>  <td>0.01617</td>
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    0.2147</td> <td>    0.061</td> <td>    3.507</td> <td> 0.000</td> <td>    0.095</td> <td>    0.335</td>
</tr>
<tr>
  <th>Lag2</th>      <td>    0.0628</td> <td>    0.026</td> <td>    2.382</td> <td> 0.017</td> <td>    0.011</td> <td>    0.114</td>
</tr>
</table>




```python
predictions = np.where(results.predict(weekly) > .5, 1, 0)
```

(c) Compute the confusion matrix and overall fraction of correct
predictions. Explain what the confusion matrix is telling you
about the types of mistakes made by logistic regression.


```python
confusion_matrix(weekly['Direction'], predictions)
```




    array([[ 33, 451],
           [ 26, 579]])




```python
451 / 1030
```




    0.43786407766990293




```python
weekly['Direction'].mean()
```




    0.5555555555555556



(d) Fit the logistic regression model using a training data period
from 1990 to 2008, with Lag2 as the only predictor. Compute the
confusion matrix and the overall fraction of correct predictions
for the held out data (that is, the data from 2009 and 2010).


```python
year_bool = weekly['Year'] < 2009
weekly['ones'] = 1
X_train = weekly[year_bool][['ones', 'Lag2']].values
X_test = weekly[~year_bool][['ones', 'Lag2']].values
y_train = weekly[year_bool]['Direction'].values
y_test = weekly[~year_bool]['Direction'].values
```


```python
lr =  LogisticRegression()
lr.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)




```python
confusion_matrix(y_test, lr.predict(X_test))
```




    array([[ 9, 34],
           [ 5, 56]])



(e) Repeat (d) using LDA.


```python
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
confusion_matrix(y_test, lda.predict(X_test))
```




    array([[ 9, 34],
           [ 5, 56]])



(f) Repeat (d) using QDA.


```python
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
confusion_matrix(y_test, qda.predict(X_test))
```

    /usr/local/lib/python3.6/dist-packages/sklearn/discriminant_analysis.py:691: UserWarning: Variables are collinear
      warnings.warn("Variables are collinear")
    /usr/local/lib/python3.6/dist-packages/sklearn/discriminant_analysis.py:715: RuntimeWarning: divide by zero encountered in power
      X2 = np.dot(Xm, R * (S ** (-0.5)))
    /usr/local/lib/python3.6/dist-packages/sklearn/discriminant_analysis.py:715: RuntimeWarning: invalid value encountered in multiply
      X2 = np.dot(Xm, R * (S ** (-0.5)))
    /usr/local/lib/python3.6/dist-packages/sklearn/discriminant_analysis.py:718: RuntimeWarning: divide by zero encountered in log
      u = np.asarray([np.sum(np.log(s)) for s in self.scalings_])
    




    array([[43,  0],
           [61,  0]])



(g) Repeat (d) using KNN with K = 1


```python
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
confusion_matrix(y_test, knn.predict(X_test))
```




    array([[21, 22],
           [30, 31]])



(h) Experiment with different combinations of predictors, including possible transformations and interactions, for each of the
methods. Report the variables, method, and associated confusion matrix that appears to provide the best results on the held
out data. Note that you should also experiment with values for
K in the KNN classifier.


```python
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)
confusion_matrix(y_test, knn.predict(X_test))
```




    array([[20, 23],
           [20, 41]])




```python
results = smf.logit('Direction ~ np.power(Lag5, 2)', data=weekly).fit()
results.summary()
```

    Optimization terminated successfully.
             Current function value: 0.686956
             Iterations 4
    




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>Direction</td>    <th>  No. Observations:  </th>  <td>  1089</td>  
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td>  1087</td>  
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>     1</td>  
</tr>
<tr>
  <th>Date:</th>            <td>Wed, 23 Sep 2020</td> <th>  Pseudo R-squ.:     </th> <td>8.318e-06</td>
</tr>
<tr>
  <th>Time:</th>                <td>07:35:15</td>     <th>  Log-Likelihood:    </th> <td> -748.09</td> 
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -748.10</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th>  <td>0.9112</td>  
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>         <td>    0.2256</td> <td>    0.065</td> <td>    3.475</td> <td> 0.001</td> <td>    0.098</td> <td>    0.353</td>
</tr>
<tr>
  <th>np.power(Lag5, 2)</th> <td>   -0.0004</td> <td>    0.004</td> <td>   -0.112</td> <td> 0.911</td> <td>   -0.008</td> <td>    0.007</td>
</tr>
</table>




```python
results = smf.logit('Direction ~ np.power(Volume, 2)', data=weekly).fit()
results.summary()
```

    Optimization terminated successfully.
             Current function value: 0.686884
             Iterations 4
    




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>Direction</td>    <th>  No. Observations:  </th>  <td>  1089</td>  
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td>  1087</td>  
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>     1</td>  
</tr>
<tr>
  <th>Date:</th>            <td>Wed, 23 Sep 2020</td> <th>  Pseudo R-squ.:     </th> <td>0.0001136</td>
</tr>
<tr>
  <th>Time:</th>                <td>07:35:15</td>     <th>  Log-Likelihood:    </th> <td> -748.02</td> 
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -748.10</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th>  <td>0.6801</td>  
</tr>
</table>
<table class="simpletable">
<tr>
           <td></td>              <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>           <td>    0.2359</td> <td>    0.068</td> <td>    3.448</td> <td> 0.001</td> <td>    0.102</td> <td>    0.370</td>
</tr>
<tr>
  <th>np.power(Volume, 2)</th> <td>   -0.0024</td> <td>    0.006</td> <td>   -0.413</td> <td> 0.680</td> <td>   -0.014</td> <td>    0.009</td>
</tr>
</table>




```python
results = smf.logit('Direction ~ Volume * Lag3', data=weekly).fit()
results.summary()
```

    Optimization terminated successfully.
             Current function value: 0.686505
             Iterations 4
    




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>Direction</td>    <th>  No. Observations:  </th>  <td>  1089</td>  
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td>  1085</td>  
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>     3</td>  
</tr>
<tr>
  <th>Date:</th>            <td>Wed, 23 Sep 2020</td> <th>  Pseudo R-squ.:     </th> <td>0.0006650</td>
</tr>
<tr>
  <th>Time:</th>                <td>07:35:15</td>     <th>  Log-Likelihood:    </th> <td> -747.60</td> 
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -748.10</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th>  <td>0.8025</td>  
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>   <td>    0.2631</td> <td>    0.084</td> <td>    3.134</td> <td> 0.002</td> <td>    0.099</td> <td>    0.428</td>
</tr>
<tr>
  <th>Volume</th>      <td>   -0.0235</td> <td>    0.036</td> <td>   -0.649</td> <td> 0.516</td> <td>   -0.095</td> <td>    0.048</td>
</tr>
<tr>
  <th>Lag3</th>        <td>   -0.0188</td> <td>    0.039</td> <td>   -0.482</td> <td> 0.630</td> <td>   -0.095</td> <td>    0.058</td>
</tr>
<tr>
  <th>Volume:Lag3</th> <td>   -0.0007</td> <td>    0.011</td> <td>   -0.067</td> <td> 0.946</td> <td>   -0.022</td> <td>    0.020</td>
</tr>
</table>



# 3. Now, lets develop a model to predict whether a given car gets high or low gas mileage based on the Auto data set.


```python
auto = pd.read_csv('https://raw.githubusercontent.com/nikbearbrown/Google_Colab/master/data/auto.csv')
auto.head()
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
    </tr>
  </tbody>
</table>
</div>



(a) Create a binary variable, mpg01, that contains a 1 if mpg contains
a value above its median, and a 0 if mpg contains a value below
its median. You can compute the median using the median()
function.

Note - you can use the data.frame() function to create a single data set containing both mpg01 and the other Auto variables.


```python
auto['mpg01'] = np.where(auto['mpg'] > auto['mpg'].median(), 1, 0)
auto.head()
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
      <th>name</th>
      <th>mpg01</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
auto.corr()
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
      <th>mpg01</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mpg</th>
      <td>1.000000</td>
      <td>-0.777618</td>
      <td>-0.805127</td>
      <td>-0.778427</td>
      <td>-0.832244</td>
      <td>0.423329</td>
      <td>0.580541</td>
      <td>0.565209</td>
      <td>0.836939</td>
    </tr>
    <tr>
      <th>cylinders</th>
      <td>-0.777618</td>
      <td>1.000000</td>
      <td>0.950823</td>
      <td>0.842983</td>
      <td>0.897527</td>
      <td>-0.504683</td>
      <td>-0.345647</td>
      <td>-0.568932</td>
      <td>-0.759194</td>
    </tr>
    <tr>
      <th>displacement</th>
      <td>-0.805127</td>
      <td>0.950823</td>
      <td>1.000000</td>
      <td>0.897257</td>
      <td>0.932994</td>
      <td>-0.543800</td>
      <td>-0.369855</td>
      <td>-0.614535</td>
      <td>-0.753477</td>
    </tr>
    <tr>
      <th>horsepower</th>
      <td>-0.778427</td>
      <td>0.842983</td>
      <td>0.897257</td>
      <td>1.000000</td>
      <td>0.864538</td>
      <td>-0.689196</td>
      <td>-0.416361</td>
      <td>-0.455171</td>
      <td>-0.667053</td>
    </tr>
    <tr>
      <th>weight</th>
      <td>-0.832244</td>
      <td>0.897527</td>
      <td>0.932994</td>
      <td>0.864538</td>
      <td>1.000000</td>
      <td>-0.416839</td>
      <td>-0.309120</td>
      <td>-0.585005</td>
      <td>-0.757757</td>
    </tr>
    <tr>
      <th>acceleration</th>
      <td>0.423329</td>
      <td>-0.504683</td>
      <td>-0.543800</td>
      <td>-0.689196</td>
      <td>-0.416839</td>
      <td>1.000000</td>
      <td>0.290316</td>
      <td>0.212746</td>
      <td>0.346822</td>
    </tr>
    <tr>
      <th>year</th>
      <td>0.580541</td>
      <td>-0.345647</td>
      <td>-0.369855</td>
      <td>-0.416361</td>
      <td>-0.309120</td>
      <td>0.290316</td>
      <td>1.000000</td>
      <td>0.181528</td>
      <td>0.429904</td>
    </tr>
    <tr>
      <th>origin</th>
      <td>0.565209</td>
      <td>-0.568932</td>
      <td>-0.614535</td>
      <td>-0.455171</td>
      <td>-0.585005</td>
      <td>0.212746</td>
      <td>0.181528</td>
      <td>1.000000</td>
      <td>0.513698</td>
    </tr>
    <tr>
      <th>mpg01</th>
      <td>0.836939</td>
      <td>-0.759194</td>
      <td>-0.753477</td>
      <td>-0.667053</td>
      <td>-0.757757</td>
      <td>0.346822</td>
      <td>0.429904</td>
      <td>0.513698</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



(b) Which of the other features seem most likely to be useful in predicting mpg01?


```python
X = auto[['cylinders', 'origin']].values
y = auto['mpg01'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```


```python
lr = LogisticRegression()
lr.fit(X_train, y_train)
accuracy_score(y_test, lr.predict(X_test))
```




    0.9285714285714286




```python
X = auto[['cylinders', 'origin', 'year', 'acceleration']].values
y = auto['mpg01'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```


```python
# slightly higher with more variables
lr = LogisticRegression()
lr.fit(X_train, y_train)
accuracy_score(y_test, lr.predict(X_test))
```




    0.8979591836734694




```python
lr = LogisticRegression(C=.01)
lr.fit(X_train, y_train)
accuracy_score(y_test, lr.predict(X_test))
```




    0.8367346938775511



(c)  Perform LDA on the training data in order to predict mpg01
using the variables that seemed most associated with mpg01 in
(b). What is the test accuracy of the model obtained?


```python
X = auto[auto.columns[1:-1].difference(['name'])].values
y = auto['mpg01'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
accuracy_score(y_test, lda.predict(X_test))
```




    0.9489795918367347



(d) Perform QDA on the training data in order to predict mpg01
using the variables that seemed most associated with mpg01 in
(b). What is the test accuracy of the model obtained?


```python
X = auto[auto.columns[1:-1].difference(['name'])].values
y = auto['mpg01'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
accuracy_score(y_test, qda.predict(X_test))
```




    0.9489795918367347



(e) Perform KNN on the training data, with several values of K, in
order to predict mpg01. Use only the variables that seemed most
associated with mpg01 in (b). What test accuracies do you obtain?
Which value of K seems to perform the best on this data set?


```python
X = auto[auto.columns[1:-1].difference(['name'])].values
y = auto['mpg01'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```


```python
for k in range(1, 51):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, knn.predict(X_test))
    print('With K={} accuracy is {:.3f}'.format(k, accuracy))
```

    With K=1 accuracy is 0.878
    With K=2 accuracy is 0.888
    With K=3 accuracy is 0.898
    With K=4 accuracy is 0.898
    With K=5 accuracy is 0.867
    With K=6 accuracy is 0.898
    With K=7 accuracy is 0.888
    With K=8 accuracy is 0.908
    With K=9 accuracy is 0.908
    With K=10 accuracy is 0.898
    With K=11 accuracy is 0.908
    With K=12 accuracy is 0.908
    With K=13 accuracy is 0.908
    With K=14 accuracy is 0.908
    With K=15 accuracy is 0.898
    With K=16 accuracy is 0.898
    With K=17 accuracy is 0.888
    With K=18 accuracy is 0.918
    With K=19 accuracy is 0.918
    With K=20 accuracy is 0.918
    With K=21 accuracy is 0.908
    With K=22 accuracy is 0.908
    With K=23 accuracy is 0.908
    With K=24 accuracy is 0.918
    With K=25 accuracy is 0.908
    With K=26 accuracy is 0.918
    With K=27 accuracy is 0.918
    With K=28 accuracy is 0.918
    With K=29 accuracy is 0.918
    With K=30 accuracy is 0.918
    With K=31 accuracy is 0.918
    With K=32 accuracy is 0.918
    With K=33 accuracy is 0.908
    With K=34 accuracy is 0.908
    With K=35 accuracy is 0.898
    With K=36 accuracy is 0.908
    With K=37 accuracy is 0.898
    With K=38 accuracy is 0.898
    With K=39 accuracy is 0.898
    With K=40 accuracy is 0.898
    With K=41 accuracy is 0.898
    With K=42 accuracy is 0.898
    With K=43 accuracy is 0.898
    With K=44 accuracy is 0.908
    With K=45 accuracy is 0.898
    With K=46 accuracy is 0.908
    With K=47 accuracy is 0.898
    With K=48 accuracy is 0.898
    With K=49 accuracy is 0.898
    With K=50 accuracy is 0.898
    

# 4. Write a function, power(x,a), allows you to pass any two numbers, x and a, and prints out the value of x^a.


```python
power = lambda x, a: x ** a
```


```python
power(3, 8)
```




    6561



# 5. Create a function, PlotPower(), that allows you to create a plot of x against x^a for a fixed a and for a range of values of x.


```python
n = 100
plt.plot(range(n), [power(x, 2) for x in range(n)])
```




    [<matplotlib.lines.Line2D at 0x7fc9beae0588>]




    
![png](output_108_1.png)
    



```python
def plot_power(rng, p):
    plt.plot(rng, [power(x, p) for x in rng])
```


```python
plot_power(range(3,14), 3)
```


    
![png](output_110_0.png)
    


# 6. Using the Boston data set, fit classification models in order to predict whether a given suburb has a crime rate above or below the median. Explore logistic regression, LDA, and KNN models using various subsets of the predictors. Describe your findings.


```python
boston = pd.read_csv('https://raw.githubusercontent.com/nikbearbrown/Google_Colab/master/data/boston.csv')
boston['crim01'] = np.where(boston['crim'] > boston['crim'].median(), 1, 0)
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
      <th>crim01</th>
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
      <td>0</td>
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
      <td>0</td>
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
      <td>0</td>
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
      <td>0</td>
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
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = boston.iloc[:,1:-1].values
y = boston['crim01'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
accuracy_score(y_test, qda.predict(X_test))
```




    0.905511811023622




```python
X = boston.iloc[:,1:-1].values
y = boston['crim01'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
accuracy_score(y_test, knn.predict(X_test))
```




    0.8740157480314961




```python
X = boston.iloc[:,1:-1].values
y = boston['crim01'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

lr = LogisticRegression(C=1)
lr.fit(X_train, y_train)
accuracy_score(y_test, lr.predict(X_test))
```

    /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    




    0.8503937007874016


