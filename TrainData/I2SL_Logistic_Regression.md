## Logistic Regression

[Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression), or logit regression, or is a regression model where the outcome variable is categorical. Often this is used when the variable is binary (e.g. yes/no, survived/dead, pass/fail, etc.)

Logistic regression measures the relationship between the categorical response variable and one or more predictor variables by estimating probabilities.


```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.util.testing as tm
from scipy import stats
import seaborn as sns
import warnings
import random
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn import metrics
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression

from datetime import datetime
random.seed(datetime.now())
warnings.filterwarnings('ignore')

# Make plots larger
plt.rcParams['figure.figsize'] = (10, 6)
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:5: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      """
    


```python
#  the number of hours each student spent studying, and whether they passed (1) or failed (0).
url = 'https://raw.githubusercontent.com/nikbearbrown/Google_Colab/master/data/passing_an_exam_versus_hours_of_study.csv'
hours_pass = pd.read_csv(url)
hours_pass.head()
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
      <th>Hours</th>
      <th>Pass</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.75</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.25</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.50</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Can we fit a line to this data?

Can we fit a line to this data rather than a logistic function?


```python
sns.lmplot(x='Hours', y='Pass', data=hours_pass, ci=None)
```




    <seaborn.axisgrid.FacetGrid at 0x7fbdd4f22190>




    
![png](output_4_1.png)
    


The reason for using logistic regression for this problem is that the
dependent variable pass/fail represented by “1” and “0” are not
cardinal numbers. If the problem was changed so that pass/fail was
replaced with the grade 0–100 (cardinal numbers), then simple
regression analysis could be used.

The table shows the number of hours each student spent studying, and
whether they passed (1) or failed (0).

| Hours |0.50|0.75|1.00|1.25|1.50|1.75|2.00|2.25|2.50|2.75|3.00|3.25|3.50|4.00|4.25|4.50|4.75|5.00|5.50|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pass | 0  |  0 |  0 |  0 | 0  |  1 |  0 |  1 |  0 |  1 | 0  | 1  | 0  | 1  | 1  |   1|  1 |  1 |  1 |


The graph shows the probability of passing the exam versus the number of
hours studying, with the logistic regression curve fitted to the data.

![Graph of a logistic regression curve showing probability of passing an
exam versus hours studying](http://nikbearbrown.com/YouTube/MachineLearning/IMG/Exam_pass_logistic_curve.jpg)


The logistic regression analysis gives the following output.


|   | Coefficient  | Std.Error  | z-value  | P-value (Wald)  |
|-----------|-----------|-----------|-----------|-----------|
|Intercept   | −4.0777  | 1.7610  | −2.316  | 0.0206  |
| Hours   | 1.5046   | 0.6287  | 2.393   | 0.0167  |



The output indicates that hours studying is significantly associated
with the probability of passing the exam ($p = 0.0167$, [Wald test](https://en.wikipedia.org/wiki/Wald_test)).
The output also provides the coefficients for
$\text{Intercept} = - 4.0777$ and $\text{Hours} = 1.5046$. These
coefficients are entered in the logistic regression equation to estimate
the probability of passing the exam:

$\text{Probability of passing exam} = \frac{1}{1 + \text{exp} \left( - \left( 1.5046 \cdot \text{Hours} - 4.0777 \right) \right) }$

For example, for a student who studies 2 hours, entering the value
$\text{Hours} = 2$ in the equation gives the estimated probability of
passing the exam of 0.26:

$\text{Probability of passing exam} = \frac{1}{1 + \text{exp}\left( - \left( 1.5046 \cdot 2 - 4.0777 \right) \right) } = 0.26$

Similarly, for a student who studies 4 hours, the estimated probability
of passing the exam is 0.87:

$\text{Probability of passing exam} = \frac{1}{1 + \text{exp}\left( - \left( 1.5046 \cdot 4 -  4.0777 \right) \right) } = 0.87$

This table shows the probability of passing the exam for several values
of hours studying.

| Hours of study  | Probability of passing exam  |
|----------------|-----------------------------|
| 1  | 0.07  |
| 2  | 0.26  |
| 3  | 0.61  |
| 4  | 0.87  |
| 5  | 0.97  |


The **Wald test** is a parametric statistical test named after the
Hungarian statistician Abraham Wald. Whenever a relationship within
or between data items can be expressed as a statistical model with
parameters to be estimated from a sample, the Wald test can be used to
test the true value of the parameter based on the sample estimate.

Suppose an economist, who has data on social class and shoe size,
wonders whether social class is associated with shoe size. Say $\theta$
is the average increase in shoe size for upper-class people compared to
middle-class people: then the Wald test can be used to test whether
$\theta$ is 0 (in which case social class has no association with shoe
size) or non-zero (shoe size varies between social classes). Here,
$\theta$, the hypothetical difference in shoe sizes between upper and
middle-class people in the whole population, is a parameter. An estimate
of $\theta$ might be the difference in shoe size between upper and
middle-class people in the sample. In the Wald test, the economist uses
the estimate and an estimate of variability (see below) to draw
conclusions about the unobserved true $\theta$. Or, for a medical
example, suppose smoking multiplies the risk of lung cancer by some
number *R*: then the Wald test can be used to test whether *R* = 1 (i.e.
there is no effect of smoking) or is greater (or less) than 1 (i.e.
smoking alters risk).

A Wald test can be used in a great variety of different models including
models for [dichotomous] variables and models for [continuous
variables].

Mathematical details
--------------------

Under the Wald statistical test, the [maximum likelihood](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) estimate
$\hat\theta$ of the parameter(s) of interest $\theta$ is compared with
the proposed value $\theta_0$, with the assumption that the difference
between the two will be approximately normally distributed. Typically
the square of the difference is compared to a [chi-squared
distribution](https://en.wikipedia.org/wiki/Chi-squared_distribution).

### Test on a single parameter

In the univariate case, the Wald statistic is

$$\frac{ ( \widehat{ \theta}-\theta_0 )^2 }{\operatorname{var}(\hat \theta )}$$

which is compared against a [chi-squared distribution](https://en.wikipedia.org/wiki/Chi-squared_distribution).

Alternatively, the difference can be compared to a normal
distribution. In this case the test statistic is

$$\frac{\widehat{\theta}-\theta_0}{\operatorname{se}(\hat\theta)}$$

where $\operatorname{se}(\widehat\theta)$ is the [standard error](https://en.wikipedia.org/wiki/Standard_error) of the
maximum likelihood estimate (MLE). A reasonable estimate of the standard
error for the MLE can be given by <math> \\frac

## Logistic model

Let us try to understand logistic regression by considering a logistic
model with given parameters, then seeing how the coefficients can be
estimated from data. Consider a model with two predictors, $x_1$ and
$x_2$, and one binary (Bernoulli) response variable $Y$, which we denote
$p=P(Y=1)$. We assume a [linear relationship] between the predictor
variables and the [log-odds] of the event that $Y=1$. This linear
relationship can be written in the following mathematical form (where
*ℓ* is the log-odds, $b$ is the base of the logarithm, and $\beta_i$ are
parameters of the model):

$$\ell = \log_b \frac{p}{1-p} = \beta_0 + \beta_1 x_1 + \beta_2 x_2$$

We can recover the [odds] by exponentiating the log-odds:

$$\frac{p}{1-p} = b^{\beta_0 + \beta_1 x_1 + \beta_2 x_2}$$.

By simple algebraic manipulation, the probability that $Y=1$ is

$$p = \frac{b^{\beta_0 + \beta_1 x_1 + \beta_2 x_2}}{b^{\beta_0 + \beta_1 x_1 + \beta_2 x_2} + 1} = \frac{1}{1 + b^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2)}}$$.

The above formula shows that once $\beta_i$ are fixed, we can easily
compute either the log-odds that $Y=1$ for a given observation, or the
probability that $Y=1$ for a given observation. The main use-case of a
logistic model is to be given an observation $(x_1,x_2)$, and estimate
the probability $p$ that $Y=1$. In most applications, the base $b$ of
the logarithm is usually taken to be *[e]*. However in some cases it can
be easier to communicate results by working in base 2, or base 10.


```python
p = np.linspace(0,1, num=100)
l = lambda p: np.log(p/(1-p))
ax=plt.gca()
ax.plot(p, l(p))
ax.axvline(0.5, color='purple', alpha=0.5, ls=':')
ax.grid(True)
ax.set_xlabel('$P$')
ax.set_ylabel('$L$')
ax.text(0.1, 3, "$\\log \\frac {P(x)} {1-P(x)}$");
```


    
![png](output_7_0.png)
    


## Logistic function (S-curve)

A *[logistic function](https://en.wikipedia.org/wiki/Logistic_function)* or *logistic curve* is a common "S" shape curve, with equation:

$$f(x) = \frac{1}{1 + \mathrm e^{-x}} $$

$$f(x) = \frac{L}{1 + \mathrm e^{-k(x-x_0)}} $$

where  

* $e$ = the natural logarithm base $e$ (also known as $e$ or Euler's number),   
* $x_0$ = the $x$-value of the sigmoid's midpoint,    
* $L$ = the curve's maximum value, and     
* $k$ = the steepness of the curve.    

For values of $x$ in the range of real number's from $-\infty$ to $\infty$, the S-curve shown on the right is obtained (with the graph of $f$ approaching $L$ as $x$ approaches $\infty$ and approaching zero as $x$ approaches $-\infty$).


```python
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
support = np.linspace(-6, 6, 1000)
ax.plot(support, stats.logistic.cdf(support), 'r-', label='Logistic')
ax.plot(support, stats.norm.cdf(support), label='Probit')
ax.set_xlabel('$x$')
ax.set_ylabel('$P(x)$')
ax.legend();
```


    
![png](output_9_0.png)
    


## Logistic Regression

**Linear regression:** continuous response is modeled as a linear combination of the features:

$$y = \beta_0 + \beta_1x + \varepsilon_i$$

**Logistic regression:** log-odds of a categorical response being "true" (1) is modeled as a linear combination of the features:

$$\ln \left({p\over 1-p}\right) = \beta_0 + \beta_1x + \varepsilon_i$$

This is called the **logit function**.

Probability is sometimes written as $\pi$:

$$\ln \left({\pi\over 1-\pi}\right) = \beta_0 + \beta_1x + \varepsilon_i$$

The equation can be rearranged into the **logistic function**:

$$\pi = \frac{e^{\beta_0 + \beta_1x}} {1 + e^{\beta_0 + \beta_1x}}$$

In other words:

- Logistic regression outputs the **probabilities of a specific class**
- Those probabilities can be converted into **class predictions**

The **logistic function** has some nice properties:

- Takes on an "s" shape
- Output is bounded by 0 and 1


```python
# Adding intercept manually
hours_pass['intercept'] = 1.0
hours_pass.head()
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
      <th>Hours</th>
      <th>Pass</th>
      <th>intercept</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.50</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.75</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.00</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.25</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.50</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pred=['Hours','intercept']
pred
```




    ['Hours', 'intercept']




```python
hours_pass_model = sm.Logit(hours_pass['Pass'], hours_pass[pred]).fit()
hours_pass_model.summary()
```

    Optimization terminated successfully.
             Current function value: 0.539410
             Iterations 6
    




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Pass</td>       <th>  No. Observations:  </th>  <td>    23</td> 
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td>    21</td> 
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>     1</td> 
</tr>
<tr>
  <th>Date:</th>            <td>Wed, 07 Apr 2021</td> <th>  Pseudo R-squ.:     </th>  <td>0.2207</td> 
</tr>
<tr>
  <th>Time:</th>                <td>04:22:04</td>     <th>  Log-Likelihood:    </th> <td> -12.406</td>
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -15.921</td>
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th> <td>0.008023</td>
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Hours</th>     <td>    0.7632</td> <td>    0.334</td> <td>    2.288</td> <td> 0.022</td> <td>    0.109</td> <td>    1.417</td>
</tr>
<tr>
  <th>intercept</th> <td>   -2.5161</td> <td>    1.169</td> <td>   -2.152</td> <td> 0.031</td> <td>   -4.808</td> <td>   -0.225</td>
</tr>
</table>



## Discusion  

 A. Write an equation that describes the model.  


 $$\ln \left({p\over 1-p}\right) = \beta_0 + \beta_1*Hours + \varepsilon_i$$

 or alternatively

 $$logit \left({p}\right) = \beta_0 + \beta_1*Hours + \varepsilon_i$$

  or alternatively

 $$logit \left({p}\right) = -2.5161 + 0.7632*Hours + \varepsilon_i$$

 B. Is the coefficient _Hours_ significant? How does one interpret the meaning of its value?

The coefficient _Hours_ is significant as the z-score is 2.3 SD above the mean. The p-value is 0.022 which is significant at a 0.05 level but not at a 0.01 level.  

The coefficient for _Hours_  is the difference in the log odds.  In other words, for a one hour increase in studying, the expected change in log odds is 0.7632.  
.

 C. Is the coefficient _intercept_ significant? How does one interpret the meaning of its value?

The coefficient _intercept is significant as the z-score is 2.1 SD below the mean. The p-value is 0.031 which is significant at a 0.05 level but not at a 0.01 level.  

In this case, the estimated coefficient for the intercept is the log odds (	-2.5161) of a student passing who has spent zero hours studying.   

D. Calculate the increase in odds of passing by studying four hours rather than three.  

The coefficient for math is the difference in the log odds.  In other words, for a one-unit increase in the math score, the expected change in log odds is 0.7632

Going from three to four hours of studying is just a one-unit increase in in the log odds (i.e. 0.7632).

To go from log odds to odds just exponentiate it.  exp(0.7632)




```python
np.exp(0.7632)
```




    2.1451296640638446




```python
b0=-2.5161
b1=0.7632
x=4
p4=np.exp(b0+b1*x)/(1+np.exp(b0+b1*x))
print(p4)
x=5
p5=np.exp(b0+b1*x)/(1+np.exp(b0+b1*x))
print(p5)
print(p5-p4)
x=6
p6=np.exp(b0+b1*x)/(1+np.exp(b0+b1*x))
print(p6)
print(p6-p5)
x=0
p0=np.exp(b0+b1*x)/(1+np.exp(b0+b1*x))
print(p0)
```

    0.6310444202632162
    0.7858181527252565
    0.1547737324620403
    0.8872646233306233
    0.10144647060536682
    0.0747371892897859
    


```python
0.7632/0.334
```




    2.2850299401197605



### coefficients of the model

Overview of the coefficients of the model, how well those coefficients fit, the overall fit quality, and several other statistical measures.

### confidence interval

LogitResults.conf_int(alpha=0.05, cols=None, method='default')
_Returns the confidence interval of the fitted parameters._

Parameters:
alpha : float, optional
_The significance level for the confidence interval. ie., The default alpha = .05 returns a 95% confidence interval._



```python
hours_pass_model.conf_int()
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
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Hours</th>
      <td>0.109432</td>
      <td>1.416992</td>
    </tr>
    <tr>
      <th>intercept</th>
      <td>-4.807533</td>
      <td>-0.224608</td>
    </tr>
  </tbody>
</table>
</div>




```python
hours_pass_model.conf_int(alpha=0.01)
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
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Hours</th>
      <td>-0.096000</td>
      <td>1.622424</td>
    </tr>
    <tr>
      <th>intercept</th>
      <td>-5.527563</td>
      <td>0.495422</td>
    </tr>
  </tbody>
</table>
</div>



### odds ratio

Take the exponential of each of the coefficients to generate the odds ratios. This tells you how a 1 unit increase or decrease in a variable affects the odds of being admitted.


```python
np.exp(hours_pass_model.params)
```




    Hours        2.145156
    intercept    0.080776
    dtype: float64




```python
# odds ratios and 95% CI
params = hours_pass_model.params
conf = hours_pass_model.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OddsRatio']
np.exp(conf)
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
      <th>2.5%</th>
      <th>97.5%</th>
      <th>OddsRatio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Hours</th>
      <td>1.115644</td>
      <td>4.124694</td>
      <td>2.145156</td>
    </tr>
    <tr>
      <th>intercept</th>
      <td>0.008168</td>
      <td>0.798829</td>
      <td>0.080776</td>
    </tr>
  </tbody>
</table>
</div>



## UCLA Admissions Data   

Variable | Description | Type of Variable
---| ---| ---
Admit| 0 = not admitted 1 = admitted | categorical
GRE | GRE score 200-800 | continuous
GPA | GPA 0-4.0 | continuous
Prestige | 1= not prestigious 2 = low prestige 3= good prestige 4= high prestige | categorical


## Admission into grad school?

Determine if there is an association between graduate school admission and the prestige of a student's undergraduate school using data from the UCLA admissions data set.

Outcome: admission into grad school.    

Predictors/covariates: Prestige, GRE, GPA  


```python
ucla = pd.read_csv("https://stats.idre.ucla.edu/stat/data/binary.csv")
ucla.head()
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
      <th>admit</th>
      <th>gre</th>
      <th>gpa</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>380</td>
      <td>3.61</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>660</td>
      <td>3.67</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>800</td>
      <td>4.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>640</td>
      <td>3.19</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>520</td>
      <td>2.93</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
ucla.describe()
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
      <th>admit</th>
      <th>gre</th>
      <th>gpa</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.317500</td>
      <td>587.700000</td>
      <td>3.389900</td>
      <td>2.48500</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.466087</td>
      <td>115.516536</td>
      <td>0.380567</td>
      <td>0.94446</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>220.000000</td>
      <td>2.260000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>520.000000</td>
      <td>3.130000</td>
      <td>2.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>580.000000</td>
      <td>3.395000</td>
      <td>2.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>660.000000</td>
      <td>3.670000</td>
      <td>3.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>800.000000</td>
      <td>4.000000</td>
      <td>4.00000</td>
    </tr>
  </tbody>
</table>
</div>




```python
ucla.std()
```




    admit      0.466087
    gre      115.516536
    gpa        0.380567
    rank       0.944460
    dtype: float64




```python
ucla.hist()
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7f1e10665a20>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f1e106ef470>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f1e10071a20>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f1e0f98d128>]],
          dtype=object)




    
![png](output_28_1.png)
    



```python
#  Tabulating whether or not someone was admitted by rank
pd.crosstab(ucla['admit'], ucla['rank'], rownames=['admit'])
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
      <th>rank</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
    <tr>
      <th>admit</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>28</td>
      <td>97</td>
      <td>93</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>54</td>
      <td>28</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>



## Dummy variables

In statistics and econometrics, particularly in regression analysis, a [dummy variable](https://en.wikipedia.org/wiki/Dummy_variable_(statistics)) (also known as an indicator variable, design variable, Boolean indicator, categorical variable, binary variable, or qualitative variable) is one that takes the value 0 or 1 to indicate the absence or presence of some categorical effect that may be expected to shift the outcome. Dummy variables are used as devices to sort data into mutually exclusive categories (such as smoker/non-smoker, etc.). For example, in econometric time series analysis, dummy variables may be used to indicate the occurrence of wars or major strikes. A dummy variable can thus be thought of as a truth value represented as a numerical value 0 or 1 (as is sometimes done in computer programming).

---------------------------------

![Graph showing wage = α~0~ + δ~0~female + α~1~education +
*U*, δ~0~ &lt; 0.](https://raw.githubusercontent.com/nikbearbrown/Google_Colab/master/img/Graph_showing_Wage_female_education.jpg)

Dummy variables are incorporated in the same way as quantitative
variables are included (as explanatory variables) in regression models.
For example, if we consider a [Mincer-type] regression model of wage
determination, wherein wages are dependent on gender (qualitative) and
years of education (quantitative):

$$\ln \text{wage} = \alpha_{0} + \delta_{0} \text{female} + \alpha_{1} \text{education} + u$$

where $u \sim N(0, \sigma^{2})$ is the [error term]. In the model,
*female* = 1 when the person is a female and *female* = 0 when the
person is male. $\delta_{0}$ can be interpreted as: the difference in
wages between females and males, holding education constant. Thus, δ~0~
helps to determine whether there is a discrimination in wages between
males and females. For example, if δ~0~&gt;0 (positive coefficient),
then women earn a higher wage than men (keeping other factors constant).
Note that the coefficients attached to the dummy variables are called
**differential intercept coefficients**. The model can be depicted
graphically as an intercept shift between females and males. In the
figure, the case δ~0~&lt;0 is shown (wherein, men earn a higher wage
than women).[^1]

Dummy variables may be extended to more complex cases. For example,
seasonal effects may be captured by creating dummy variables for each of
the seasons: $D_{1} = 1$ if the observation is for summer, and equals
zero otherwise; $D_{2}=1$ if and only if autumn, otherwise equals zero;
$D_{3}=1$ if and only if winter, otherwise equals zero; and $D_{4}=1$ if
and only if spring, otherwise equals zero. In the [panel data], [fixed
effects estimator] dummies are created for each of the units in
[cross-sectional data] (e.g. firms or countries) or periods in a [pooled
time-series]. However in such regressions either the [constant term] has
to be removed or one of the dummies has to be removed, with its
associated category becoming the base category against which the others
are assessed in order to avoid the **dummy variable trap**:

The constant term in all regression equations is a coefficient
multiplied by a regressor equal to one. When the regression is expressed
as a matrix equation, the matrix of regressors then consists of a column
of ones (the constant term), vectors of zeros and ones


```python
dummy_ranks = pd.get_dummies(ucla['rank'], prefix='rank')
dummy_ranks.head()
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
      <th>rank_1</th>
      <th>rank_2</th>
      <th>rank_3</th>
      <th>rank_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
keep = ['admit', 'gre', 'gpa']
ucla_dummy = ucla[keep].join(dummy_ranks.loc[:, 'rank_2':])
ucla_dummy.head()
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
      <th>admit</th>
      <th>gre</th>
      <th>gpa</th>
      <th>rank_2</th>
      <th>rank_3</th>
      <th>rank_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>380</td>
      <td>3.61</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>660</td>
      <td>3.67</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>800</td>
      <td>4.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>640</td>
      <td>3.19</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>520</td>
      <td>2.93</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Interpreting the Intercept in a Regression Model

$$\ln \left({p\over 1-p}\right) = \beta_0 + \beta_1x + \beta_2x + \beta_3x + \beta_4x .... + \varepsilon_i$$


The intercept $\beta_0$ is the expected mean value when all $\beta_ix = 0$.


```python
# Adding intercept manually
ucla_dummy['intercept'] = 1.0
ucla_dummy.head()
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
      <th>admit</th>
      <th>gre</th>
      <th>gpa</th>
      <th>rank_2</th>
      <th>rank_3</th>
      <th>rank_4</th>
      <th>intercept</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>380</td>
      <td>3.61</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>660</td>
      <td>3.67</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>800</td>
      <td>4.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>640</td>
      <td>3.19</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>520</td>
      <td>2.93</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pred = ucla_dummy.columns[1:]
pred
```




    Index(['gre', 'gpa', 'rank_2', 'rank_3', 'rank_4', 'intercept'], dtype='object')




```python
ucla_model = sm.Logit(ucla_dummy['admit'], ucla_dummy[pred]).fit()
ucla_model.summary()
```

    Optimization terminated successfully.
             Current function value: 0.573147
             Iterations 6
    




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>admit</td>      <th>  No. Observations:  </th>  <td>   400</td>  
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td>   394</td>  
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>     5</td>  
</tr>
<tr>
  <th>Date:</th>            <td>Wed, 16 Sep 2020</td> <th>  Pseudo R-squ.:     </th>  <td>0.08292</td> 
</tr>
<tr>
  <th>Time:</th>                <td>18:05:43</td>     <th>  Log-Likelihood:    </th> <td> -229.26</td> 
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -249.99</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th> <td>7.578e-08</td>
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>gre</th>       <td>    0.0023</td> <td>    0.001</td> <td>    2.070</td> <td> 0.038</td> <td>    0.000</td> <td>    0.004</td>
</tr>
<tr>
  <th>gpa</th>       <td>    0.8040</td> <td>    0.332</td> <td>    2.423</td> <td> 0.015</td> <td>    0.154</td> <td>    1.454</td>
</tr>
<tr>
  <th>rank_2</th>    <td>   -0.6754</td> <td>    0.316</td> <td>   -2.134</td> <td> 0.033</td> <td>   -1.296</td> <td>   -0.055</td>
</tr>
<tr>
  <th>rank_3</th>    <td>   -1.3402</td> <td>    0.345</td> <td>   -3.881</td> <td> 0.000</td> <td>   -2.017</td> <td>   -0.663</td>
</tr>
<tr>
  <th>rank_4</th>    <td>   -1.5515</td> <td>    0.418</td> <td>   -3.713</td> <td> 0.000</td> <td>   -2.370</td> <td>   -0.733</td>
</tr>
<tr>
  <th>intercept</th> <td>   -3.9900</td> <td>    1.140</td> <td>   -3.500</td> <td> 0.000</td> <td>   -6.224</td> <td>   -1.756</td>
</tr>
</table>




## Interpretation UCLA Admissions Data  

The output shows the coefficients, their standard errors, the z-statistic (i.e. Wald z-statistic), and the associated p-values. Both gre and gpa are statistically significant, as are the three terms for rank. The logistic regression coefficients give the change in the log odds of the outcome for a one unit increase in the predictor variable.


_gre is statistically significant_ (At .05 significance level, P>|z| < .05)

For every one unit change in gre, the log odds of admission (versus non-admission) increases by 0.0023.  Notice that the log odds increase is very small as this version of the gre goes up to 800 points. Therefore the difference between 733 and 734 gre scores is not much.  

_gpa is statistically significant_ At .05 significance level, P>|z| < .05)

For a one unit increase in gpa, the log odds of being admitted to graduate school increases by 0.8040.

The indicator variables for rank have a slightly different interpretation. The three terms for rank are statistically significant at .05 significance level, P>|z| < .05.

1= not prestigious 2 = low prestige 3= good prestige 4= high prestige

The indicator variables for rank have a slightly different interpretation.

For example, attending an undergraduate institution with rank of 2, versus an institution with a rank of 1, changes the log odds of admission by -0.6754. Attending an undergraduate institution with rank of 4, versus an institution with a rank of 1, changes the log odds of admission by -1.5515.


```python
ucla_model.conf_int()
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
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gre</th>
      <td>0.000120</td>
      <td>0.004409</td>
    </tr>
    <tr>
      <th>gpa</th>
      <td>0.153684</td>
      <td>1.454391</td>
    </tr>
    <tr>
      <th>rank_2</th>
      <td>-1.295751</td>
      <td>-0.055135</td>
    </tr>
    <tr>
      <th>rank_3</th>
      <td>-2.016992</td>
      <td>-0.663416</td>
    </tr>
    <tr>
      <th>rank_4</th>
      <td>-2.370399</td>
      <td>-0.732529</td>
    </tr>
    <tr>
      <th>intercept</th>
      <td>-6.224242</td>
      <td>-1.755716</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.exp(ucla_model.params)
```




    gre          1.002267
    gpa          2.234545
    rank_2       0.508931
    rank_3       0.261792
    rank_4       0.211938
    intercept    0.018500
    dtype: float64




```python
params = ucla_model.params
conf = ucla_model.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OddRatio']
np.exp(conf)
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
      <th>2.5%</th>
      <th>97.5%</th>
      <th>OddRatio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gre</th>
      <td>1.000120</td>
      <td>1.004418</td>
      <td>1.002267</td>
    </tr>
    <tr>
      <th>gpa</th>
      <td>1.166122</td>
      <td>4.281877</td>
      <td>2.234545</td>
    </tr>
    <tr>
      <th>rank_2</th>
      <td>0.273692</td>
      <td>0.946358</td>
      <td>0.508931</td>
    </tr>
    <tr>
      <th>rank_3</th>
      <td>0.133055</td>
      <td>0.515089</td>
      <td>0.261792</td>
    </tr>
    <tr>
      <th>rank_4</th>
      <td>0.093443</td>
      <td>0.480692</td>
      <td>0.211938</td>
    </tr>
    <tr>
      <th>intercept</th>
      <td>0.001981</td>
      <td>0.172783</td>
      <td>0.018500</td>
    </tr>
  </tbody>
</table>
</div>



## Breast Cancer Wisconsin (Diagnostic) Data Set

In the project we'll be using the _Breast Cancer Wisconsin (Diagnostic) Data Set_ to predict whether the cancer is benign or malignant.

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

This data set is from the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

This data set is also available through the UW CS ftp server: ftp ftp.cs.wisc.edu cd math-prog/cpo-dataset/machine-learn/WDBC/

Also can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

Attribute Information:

1) ID number 2) Diagnosis (M = malignant, B = benign) 3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter) b) texture (standard deviation of gray-scale values) c) perimeter d) area e) smoothness (local variation in radius lengths) f) compactness (perimeter^2 / area - 1.0) g) concavity (severity of concave portions of the contour) h) concave points (number of concave portions of the contour) i) symmetry j) fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.


```python
X=['radius_mean','radius_sd_error','radius_worst','texture_mean','texture_sd_error','texture_worst','perimeter_mean','perimeter_sd_error','perimeter_worst','area_mean','area_sd_error','area_worst','smoothness_mean','smoothness_sd_error','smoothness_worst','compactness_mean','compactness_sd_error','compactness_worst','concavity_mean','concavity_sd_error','concavity_worst','concave_points_mean','concave_points_sd_error','concave_points_worst','symmetry_mean','symmetry_sd_error','symmetry_worst','fractal_dimension_mean','fractal_dimension_sd_error','fractal_dimension_worst']
X
```




    ['radius_mean',
     'radius_sd_error',
     'radius_worst',
     'texture_mean',
     'texture_sd_error',
     'texture_worst',
     'perimeter_mean',
     'perimeter_sd_error',
     'perimeter_worst',
     'area_mean',
     'area_sd_error',
     'area_worst',
     'smoothness_mean',
     'smoothness_sd_error',
     'smoothness_worst',
     'compactness_mean',
     'compactness_sd_error',
     'compactness_worst',
     'concavity_mean',
     'concavity_sd_error',
     'concavity_worst',
     'concave_points_mean',
     'concave_points_sd_error',
     'concave_points_worst',
     'symmetry_mean',
     'symmetry_sd_error',
     'symmetry_worst',
     'fractal_dimension_mean',
     'fractal_dimension_sd_error',
     'fractal_dimension_worst']




```python
field_names = ['ID','diagnosis','radius_mean','radius_sd_error','radius_worst','texture_mean','texture_sd_error','texture_worst','perimeter_mean','perimeter_sd_error','perimeter_worst','area_mean','area_sd_error','area_worst','smoothness_mean','smoothness_sd_error','smoothness_worst','compactness_mean','compactness_sd_error','compactness_worst','concavity_mean','concavity_sd_error','concavity_worst','concave_points_mean','concave_points_sd_error','concave_points_worst','symmetry_mean','symmetry_sd_error','symmetry_worst','fractal_dimension_mean','fractal_dimension_sd_error','fractal_dimension_worst']
```


```python
breast_cancer = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None,names = field_names)
breast_cancer.head()
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
      <th>ID</th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>radius_sd_error</th>
      <th>radius_worst</th>
      <th>texture_mean</th>
      <th>texture_sd_error</th>
      <th>texture_worst</th>
      <th>perimeter_mean</th>
      <th>perimeter_sd_error</th>
      <th>perimeter_worst</th>
      <th>area_mean</th>
      <th>area_sd_error</th>
      <th>area_worst</th>
      <th>smoothness_mean</th>
      <th>smoothness_sd_error</th>
      <th>smoothness_worst</th>
      <th>compactness_mean</th>
      <th>compactness_sd_error</th>
      <th>compactness_worst</th>
      <th>concavity_mean</th>
      <th>concavity_sd_error</th>
      <th>concavity_worst</th>
      <th>concave_points_mean</th>
      <th>concave_points_sd_error</th>
      <th>concave_points_worst</th>
      <th>symmetry_mean</th>
      <th>symmetry_sd_error</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_mean</th>
      <th>fractal_dimension_sd_error</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842302</td>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>1.0950</td>
      <td>0.9053</td>
      <td>8.589</td>
      <td>153.40</td>
      <td>0.006399</td>
      <td>0.04904</td>
      <td>0.05373</td>
      <td>0.01587</td>
      <td>0.03003</td>
      <td>0.006193</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>842517</td>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>0.5435</td>
      <td>0.7339</td>
      <td>3.398</td>
      <td>74.08</td>
      <td>0.005225</td>
      <td>0.01308</td>
      <td>0.01860</td>
      <td>0.01340</td>
      <td>0.01389</td>
      <td>0.003532</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84300903</td>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>0.7456</td>
      <td>0.7869</td>
      <td>4.585</td>
      <td>94.03</td>
      <td>0.006150</td>
      <td>0.04006</td>
      <td>0.03832</td>
      <td>0.02058</td>
      <td>0.02250</td>
      <td>0.004571</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84348301</td>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>0.4956</td>
      <td>1.1560</td>
      <td>3.445</td>
      <td>27.23</td>
      <td>0.009110</td>
      <td>0.07458</td>
      <td>0.05661</td>
      <td>0.01867</td>
      <td>0.05963</td>
      <td>0.009208</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84358402</td>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>0.7572</td>
      <td>0.7813</td>
      <td>5.438</td>
      <td>94.44</td>
      <td>0.011490</td>
      <td>0.02461</td>
      <td>0.05688</td>
      <td>0.01885</td>
      <td>0.01756</td>
      <td>0.005115</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
</div>




```python
#data formating ID is a non-informative column
breast_cancer = breast_cancer.drop("ID", 1)
breast_cancer.head()
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
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>radius_sd_error</th>
      <th>radius_worst</th>
      <th>texture_mean</th>
      <th>texture_sd_error</th>
      <th>texture_worst</th>
      <th>perimeter_mean</th>
      <th>perimeter_sd_error</th>
      <th>perimeter_worst</th>
      <th>area_mean</th>
      <th>area_sd_error</th>
      <th>area_worst</th>
      <th>smoothness_mean</th>
      <th>smoothness_sd_error</th>
      <th>smoothness_worst</th>
      <th>compactness_mean</th>
      <th>compactness_sd_error</th>
      <th>compactness_worst</th>
      <th>concavity_mean</th>
      <th>concavity_sd_error</th>
      <th>concavity_worst</th>
      <th>concave_points_mean</th>
      <th>concave_points_sd_error</th>
      <th>concave_points_worst</th>
      <th>symmetry_mean</th>
      <th>symmetry_sd_error</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_mean</th>
      <th>fractal_dimension_sd_error</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>1.0950</td>
      <td>0.9053</td>
      <td>8.589</td>
      <td>153.40</td>
      <td>0.006399</td>
      <td>0.04904</td>
      <td>0.05373</td>
      <td>0.01587</td>
      <td>0.03003</td>
      <td>0.006193</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>0.5435</td>
      <td>0.7339</td>
      <td>3.398</td>
      <td>74.08</td>
      <td>0.005225</td>
      <td>0.01308</td>
      <td>0.01860</td>
      <td>0.01340</td>
      <td>0.01389</td>
      <td>0.003532</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>0.7456</td>
      <td>0.7869</td>
      <td>4.585</td>
      <td>94.03</td>
      <td>0.006150</td>
      <td>0.04006</td>
      <td>0.03832</td>
      <td>0.02058</td>
      <td>0.02250</td>
      <td>0.004571</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>0.4956</td>
      <td>1.1560</td>
      <td>3.445</td>
      <td>27.23</td>
      <td>0.009110</td>
      <td>0.07458</td>
      <td>0.05661</td>
      <td>0.01867</td>
      <td>0.05963</td>
      <td>0.009208</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>0.7572</td>
      <td>0.7813</td>
      <td>5.438</td>
      <td>94.44</td>
      <td>0.011490</td>
      <td>0.02461</td>
      <td>0.05688</td>
      <td>0.01885</td>
      <td>0.01756</td>
      <td>0.005115</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
</div>




```python
breast_cancer.groupby('diagnosis').count()
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
      <th>radius_mean</th>
      <th>radius_sd_error</th>
      <th>radius_worst</th>
      <th>texture_mean</th>
      <th>texture_sd_error</th>
      <th>texture_worst</th>
      <th>perimeter_mean</th>
      <th>perimeter_sd_error</th>
      <th>perimeter_worst</th>
      <th>area_mean</th>
      <th>area_sd_error</th>
      <th>area_worst</th>
      <th>smoothness_mean</th>
      <th>smoothness_sd_error</th>
      <th>smoothness_worst</th>
      <th>compactness_mean</th>
      <th>compactness_sd_error</th>
      <th>compactness_worst</th>
      <th>concavity_mean</th>
      <th>concavity_sd_error</th>
      <th>concavity_worst</th>
      <th>concave_points_mean</th>
      <th>concave_points_sd_error</th>
      <th>concave_points_worst</th>
      <th>symmetry_mean</th>
      <th>symmetry_sd_error</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_mean</th>
      <th>fractal_dimension_sd_error</th>
      <th>fractal_dimension_worst</th>
    </tr>
    <tr>
      <th>diagnosis</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>B</th>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
      <td>357</td>
    </tr>
    <tr>
      <th>M</th>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
    </tr>
  </tbody>
</table>
</div>




```python
breast_cancer.describe()
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
      <th>radius_mean</th>
      <th>radius_sd_error</th>
      <th>radius_worst</th>
      <th>texture_mean</th>
      <th>texture_sd_error</th>
      <th>texture_worst</th>
      <th>perimeter_mean</th>
      <th>perimeter_sd_error</th>
      <th>perimeter_worst</th>
      <th>area_mean</th>
      <th>area_sd_error</th>
      <th>area_worst</th>
      <th>smoothness_mean</th>
      <th>smoothness_sd_error</th>
      <th>smoothness_worst</th>
      <th>compactness_mean</th>
      <th>compactness_sd_error</th>
      <th>compactness_worst</th>
      <th>concavity_mean</th>
      <th>concavity_sd_error</th>
      <th>concavity_worst</th>
      <th>concave_points_mean</th>
      <th>concave_points_sd_error</th>
      <th>concave_points_worst</th>
      <th>symmetry_mean</th>
      <th>symmetry_sd_error</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_mean</th>
      <th>fractal_dimension_sd_error</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14.127292</td>
      <td>19.289649</td>
      <td>91.969033</td>
      <td>654.889104</td>
      <td>0.096360</td>
      <td>0.104341</td>
      <td>0.088799</td>
      <td>0.048919</td>
      <td>0.181162</td>
      <td>0.062798</td>
      <td>0.405172</td>
      <td>1.216853</td>
      <td>2.866059</td>
      <td>40.337079</td>
      <td>0.007041</td>
      <td>0.025478</td>
      <td>0.031894</td>
      <td>0.011796</td>
      <td>0.020542</td>
      <td>0.003795</td>
      <td>16.269190</td>
      <td>25.677223</td>
      <td>107.261213</td>
      <td>880.583128</td>
      <td>0.132369</td>
      <td>0.254265</td>
      <td>0.272188</td>
      <td>0.114606</td>
      <td>0.290076</td>
      <td>0.083946</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.524049</td>
      <td>4.301036</td>
      <td>24.298981</td>
      <td>351.914129</td>
      <td>0.014064</td>
      <td>0.052813</td>
      <td>0.079720</td>
      <td>0.038803</td>
      <td>0.027414</td>
      <td>0.007060</td>
      <td>0.277313</td>
      <td>0.551648</td>
      <td>2.021855</td>
      <td>45.491006</td>
      <td>0.003003</td>
      <td>0.017908</td>
      <td>0.030186</td>
      <td>0.006170</td>
      <td>0.008266</td>
      <td>0.002646</td>
      <td>4.833242</td>
      <td>6.146258</td>
      <td>33.602542</td>
      <td>569.356993</td>
      <td>0.022832</td>
      <td>0.157336</td>
      <td>0.208624</td>
      <td>0.065732</td>
      <td>0.061867</td>
      <td>0.018061</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6.981000</td>
      <td>9.710000</td>
      <td>43.790000</td>
      <td>143.500000</td>
      <td>0.052630</td>
      <td>0.019380</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.106000</td>
      <td>0.049960</td>
      <td>0.111500</td>
      <td>0.360200</td>
      <td>0.757000</td>
      <td>6.802000</td>
      <td>0.001713</td>
      <td>0.002252</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.007882</td>
      <td>0.000895</td>
      <td>7.930000</td>
      <td>12.020000</td>
      <td>50.410000</td>
      <td>185.200000</td>
      <td>0.071170</td>
      <td>0.027290</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.156500</td>
      <td>0.055040</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>11.700000</td>
      <td>16.170000</td>
      <td>75.170000</td>
      <td>420.300000</td>
      <td>0.086370</td>
      <td>0.064920</td>
      <td>0.029560</td>
      <td>0.020310</td>
      <td>0.161900</td>
      <td>0.057700</td>
      <td>0.232400</td>
      <td>0.833900</td>
      <td>1.606000</td>
      <td>17.850000</td>
      <td>0.005169</td>
      <td>0.013080</td>
      <td>0.015090</td>
      <td>0.007638</td>
      <td>0.015160</td>
      <td>0.002248</td>
      <td>13.010000</td>
      <td>21.080000</td>
      <td>84.110000</td>
      <td>515.300000</td>
      <td>0.116600</td>
      <td>0.147200</td>
      <td>0.114500</td>
      <td>0.064930</td>
      <td>0.250400</td>
      <td>0.071460</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13.370000</td>
      <td>18.840000</td>
      <td>86.240000</td>
      <td>551.100000</td>
      <td>0.095870</td>
      <td>0.092630</td>
      <td>0.061540</td>
      <td>0.033500</td>
      <td>0.179200</td>
      <td>0.061540</td>
      <td>0.324200</td>
      <td>1.108000</td>
      <td>2.287000</td>
      <td>24.530000</td>
      <td>0.006380</td>
      <td>0.020450</td>
      <td>0.025890</td>
      <td>0.010930</td>
      <td>0.018730</td>
      <td>0.003187</td>
      <td>14.970000</td>
      <td>25.410000</td>
      <td>97.660000</td>
      <td>686.500000</td>
      <td>0.131300</td>
      <td>0.211900</td>
      <td>0.226700</td>
      <td>0.099930</td>
      <td>0.282200</td>
      <td>0.080040</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>15.780000</td>
      <td>21.800000</td>
      <td>104.100000</td>
      <td>782.700000</td>
      <td>0.105300</td>
      <td>0.130400</td>
      <td>0.130700</td>
      <td>0.074000</td>
      <td>0.195700</td>
      <td>0.066120</td>
      <td>0.478900</td>
      <td>1.474000</td>
      <td>3.357000</td>
      <td>45.190000</td>
      <td>0.008146</td>
      <td>0.032450</td>
      <td>0.042050</td>
      <td>0.014710</td>
      <td>0.023480</td>
      <td>0.004558</td>
      <td>18.790000</td>
      <td>29.720000</td>
      <td>125.400000</td>
      <td>1084.000000</td>
      <td>0.146000</td>
      <td>0.339100</td>
      <td>0.382900</td>
      <td>0.161400</td>
      <td>0.317900</td>
      <td>0.092080</td>
    </tr>
    <tr>
      <th>max</th>
      <td>28.110000</td>
      <td>39.280000</td>
      <td>188.500000</td>
      <td>2501.000000</td>
      <td>0.163400</td>
      <td>0.345400</td>
      <td>0.426800</td>
      <td>0.201200</td>
      <td>0.304000</td>
      <td>0.097440</td>
      <td>2.873000</td>
      <td>4.885000</td>
      <td>21.980000</td>
      <td>542.200000</td>
      <td>0.031130</td>
      <td>0.135400</td>
      <td>0.396000</td>
      <td>0.052790</td>
      <td>0.078950</td>
      <td>0.029840</td>
      <td>36.040000</td>
      <td>49.540000</td>
      <td>251.200000</td>
      <td>4254.000000</td>
      <td>0.222600</td>
      <td>1.058000</td>
      <td>1.252000</td>
      <td>0.291000</td>
      <td>0.663800</td>
      <td>0.207500</td>
    </tr>
  </tbody>
</table>
</div>




```python
breast_cancer.groupby('diagnosis').median()
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
      <th>radius_mean</th>
      <th>radius_sd_error</th>
      <th>radius_worst</th>
      <th>texture_mean</th>
      <th>texture_sd_error</th>
      <th>texture_worst</th>
      <th>perimeter_mean</th>
      <th>perimeter_sd_error</th>
      <th>perimeter_worst</th>
      <th>area_mean</th>
      <th>area_sd_error</th>
      <th>area_worst</th>
      <th>smoothness_mean</th>
      <th>smoothness_sd_error</th>
      <th>smoothness_worst</th>
      <th>compactness_mean</th>
      <th>compactness_sd_error</th>
      <th>compactness_worst</th>
      <th>concavity_mean</th>
      <th>concavity_sd_error</th>
      <th>concavity_worst</th>
      <th>concave_points_mean</th>
      <th>concave_points_sd_error</th>
      <th>concave_points_worst</th>
      <th>symmetry_mean</th>
      <th>symmetry_sd_error</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_mean</th>
      <th>fractal_dimension_sd_error</th>
      <th>fractal_dimension_worst</th>
    </tr>
    <tr>
      <th>diagnosis</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>B</th>
      <td>12.200</td>
      <td>17.39</td>
      <td>78.18</td>
      <td>458.4</td>
      <td>0.09076</td>
      <td>0.07529</td>
      <td>0.03709</td>
      <td>0.02344</td>
      <td>0.1714</td>
      <td>0.061540</td>
      <td>0.2575</td>
      <td>1.1080</td>
      <td>1.8510</td>
      <td>19.630</td>
      <td>0.006530</td>
      <td>0.01631</td>
      <td>0.018400</td>
      <td>0.009061</td>
      <td>0.01909</td>
      <td>0.002808</td>
      <td>13.35</td>
      <td>22.820</td>
      <td>86.92</td>
      <td>547.4</td>
      <td>0.12540</td>
      <td>0.16980</td>
      <td>0.1412</td>
      <td>0.07431</td>
      <td>0.2687</td>
      <td>0.07712</td>
    </tr>
    <tr>
      <th>M</th>
      <td>17.325</td>
      <td>21.46</td>
      <td>114.20</td>
      <td>932.0</td>
      <td>0.10220</td>
      <td>0.13235</td>
      <td>0.15135</td>
      <td>0.08628</td>
      <td>0.1899</td>
      <td>0.061575</td>
      <td>0.5472</td>
      <td>1.1025</td>
      <td>3.6795</td>
      <td>58.455</td>
      <td>0.006209</td>
      <td>0.02859</td>
      <td>0.037125</td>
      <td>0.014205</td>
      <td>0.01770</td>
      <td>0.003739</td>
      <td>20.59</td>
      <td>28.945</td>
      <td>138.00</td>
      <td>1303.0</td>
      <td>0.14345</td>
      <td>0.35635</td>
      <td>0.4049</td>
      <td>0.18200</td>
      <td>0.3103</td>
      <td>0.08760</td>
    </tr>
  </tbody>
</table>
</div>




```python
def scaled_df(df):
    scaled = pd.DataFrame()
    for item in df:
        if item in df.select_dtypes(include=[np.float]):
            scaled[item] = ((df[item] - df[item].min()) /
            (df[item].max() - df[item].min()))
        else:
            scaled[item] = df[item]
    return scaled
breast_cancer_scaled = scaled_df(breast_cancer)
```


```python
f, ax = plt.subplots(figsize=(11, 15))
plt.title("Box Plot Breast Cancer Data Unscaled")
ax.set(xlim=(-.05, 5.0))
ax = sns.boxplot(data = breast_cancer[1:29],
  orient = 'h',
  palette = 'Set3')
```


    
![png](output_50_0.png)
    



```python
f, ax = plt.subplots(figsize=(11, 15))
plt.title("Box Plot Breast Cancer Data Scaled")
ax.set(xlim=(-.05, 1.05))
ax = sns.boxplot(data = breast_cancer_scaled[1:29],
  orient = 'h',
  palette = 'Set3')
```


    
![png](output_51_0.png)
    



```python
def rank_predictors(dat,l,f='diagnosis'):
    rank={}
    max_vals=dat.max()
    median_vals=dat.groupby(f).median()  # We are using the median as the mean is sensitive to outliers
    for p in l:
        score=np.abs((median_vals[p]['B']-median_vals[p]['M'])/max_vals[p])
        rank[p]=score
    return rank
cat_rank=rank_predictors(breast_cancer,X)
cat_rank
```




    {'area_mean': 0.00035919540229885377,
     'area_sd_error': 0.10083536373129133,
     'area_worst': 0.0011258955987717171,
     'compactness_mean': 0.09069423929098969,
     'compactness_sd_error': 0.04728535353535355,
     'compactness_worst': 0.09744269748058344,
     'concave_points_mean': 0.12363746467501009,
     'concave_points_sd_error': 0.203343949044586,
     'concave_points_worst': 0.17762106252938412,
     'concavity_mean': 0.017606079797340073,
     'concavity_sd_error': 0.031216487935656838,
     'concavity_worst': 0.20088790233074363,
     'fractal_dimension_mean': 0.3700687285223367,
     'fractal_dimension_sd_error': 0.06266947875866229,
     'fractal_dimension_worst': 0.05050602409638549,
     'perimeter_mean': 0.2677132146204311,
     'perimeter_sd_error': 0.31232604373757455,
     'perimeter_worst': 0.060855263157894794,
     'radius_mean': 0.18231945926716484,
     'radius_sd_error': 0.10361507128309573,
     'radius_worst': 0.19108753315649865,
     'smoothness_mean': 0.08318926296633303,
     'smoothness_sd_error': 0.0716064182958318,
     'smoothness_worst': 0.010295534853838727,
     'symmetry_mean': 0.08108715184186875,
     'symmetry_sd_error': 0.17632325141776936,
     'symmetry_worst': 0.21062300319488816,
     'texture_mean': 0.1893642542982807,
     'texture_sd_error': 0.07001223990208073,
     'texture_worst': 0.16519976838448178}




```python
cat_rank=sorted(cat_rank.items(), key=lambda x: x[1])
cat_rank
```




    [('area_mean', 0.00035919540229885377),
     ('area_worst', 0.0011258955987717171),
     ('smoothness_worst', 0.010295534853838727),
     ('concavity_mean', 0.017606079797340073),
     ('concavity_sd_error', 0.031216487935656838),
     ('compactness_sd_error', 0.04728535353535355),
     ('fractal_dimension_worst', 0.05050602409638549),
     ('perimeter_worst', 0.060855263157894794),
     ('fractal_dimension_sd_error', 0.06266947875866229),
     ('texture_sd_error', 0.07001223990208073),
     ('smoothness_sd_error', 0.0716064182958318),
     ('symmetry_mean', 0.08108715184186875),
     ('smoothness_mean', 0.08318926296633303),
     ('compactness_mean', 0.09069423929098969),
     ('compactness_worst', 0.09744269748058344),
     ('area_sd_error', 0.10083536373129133),
     ('radius_sd_error', 0.10361507128309573),
     ('concave_points_mean', 0.12363746467501009),
     ('texture_worst', 0.16519976838448178),
     ('symmetry_sd_error', 0.17632325141776936),
     ('concave_points_worst', 0.17762106252938412),
     ('radius_mean', 0.18231945926716484),
     ('texture_mean', 0.1893642542982807),
     ('radius_worst', 0.19108753315649865),
     ('concavity_worst', 0.20088790233074363),
     ('concave_points_sd_error', 0.203343949044586),
     ('symmetry_worst', 0.21062300319488816),
     ('perimeter_mean', 0.2677132146204311),
     ('perimeter_sd_error', 0.31232604373757455),
     ('fractal_dimension_mean', 0.3700687285223367)]




```python
# Take the top predictors based on median difference
ranked_predictors=[]
for f in cat_rank[18:]:
    ranked_predictors.append(f[0])
ranked_predictors
```




    ['texture_worst',
     'symmetry_sd_error',
     'concave_points_worst',
     'radius_mean',
     'texture_mean',
     'radius_worst',
     'concavity_worst',
     'concave_points_sd_error',
     'symmetry_worst',
     'perimeter_mean',
     'perimeter_sd_error',
     'fractal_dimension_mean']




```python
Xdf = breast_cancer_scaled[X]
#setting target
y = breast_cancer_scaled["diagnosis"]
Xdf.head()
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
      <th>radius_mean</th>
      <th>radius_sd_error</th>
      <th>radius_worst</th>
      <th>texture_mean</th>
      <th>texture_sd_error</th>
      <th>texture_worst</th>
      <th>perimeter_mean</th>
      <th>perimeter_sd_error</th>
      <th>perimeter_worst</th>
      <th>area_mean</th>
      <th>area_sd_error</th>
      <th>area_worst</th>
      <th>smoothness_mean</th>
      <th>smoothness_sd_error</th>
      <th>smoothness_worst</th>
      <th>compactness_mean</th>
      <th>compactness_sd_error</th>
      <th>compactness_worst</th>
      <th>concavity_mean</th>
      <th>concavity_sd_error</th>
      <th>concavity_worst</th>
      <th>concave_points_mean</th>
      <th>concave_points_sd_error</th>
      <th>concave_points_worst</th>
      <th>symmetry_mean</th>
      <th>symmetry_sd_error</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_mean</th>
      <th>fractal_dimension_sd_error</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.521037</td>
      <td>0.022658</td>
      <td>0.545989</td>
      <td>0.363733</td>
      <td>0.593753</td>
      <td>0.792037</td>
      <td>0.703140</td>
      <td>0.731113</td>
      <td>0.686364</td>
      <td>0.605518</td>
      <td>0.356147</td>
      <td>0.120469</td>
      <td>0.369034</td>
      <td>0.273811</td>
      <td>0.159296</td>
      <td>0.351398</td>
      <td>0.135682</td>
      <td>0.300625</td>
      <td>0.311645</td>
      <td>0.183042</td>
      <td>0.620776</td>
      <td>0.141525</td>
      <td>0.668310</td>
      <td>0.450698</td>
      <td>0.601136</td>
      <td>0.619292</td>
      <td>0.568610</td>
      <td>0.912027</td>
      <td>0.598462</td>
      <td>0.418864</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.643144</td>
      <td>0.272574</td>
      <td>0.615783</td>
      <td>0.501591</td>
      <td>0.289880</td>
      <td>0.181768</td>
      <td>0.203608</td>
      <td>0.348757</td>
      <td>0.379798</td>
      <td>0.141323</td>
      <td>0.156437</td>
      <td>0.082589</td>
      <td>0.124440</td>
      <td>0.125660</td>
      <td>0.119387</td>
      <td>0.081323</td>
      <td>0.046970</td>
      <td>0.253836</td>
      <td>0.084539</td>
      <td>0.091110</td>
      <td>0.606901</td>
      <td>0.303571</td>
      <td>0.539818</td>
      <td>0.435214</td>
      <td>0.347553</td>
      <td>0.154563</td>
      <td>0.192971</td>
      <td>0.639175</td>
      <td>0.233590</td>
      <td>0.222878</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.601496</td>
      <td>0.390260</td>
      <td>0.595743</td>
      <td>0.449417</td>
      <td>0.514309</td>
      <td>0.431017</td>
      <td>0.462512</td>
      <td>0.635686</td>
      <td>0.509596</td>
      <td>0.211247</td>
      <td>0.229622</td>
      <td>0.094303</td>
      <td>0.180370</td>
      <td>0.162922</td>
      <td>0.150831</td>
      <td>0.283955</td>
      <td>0.096768</td>
      <td>0.389847</td>
      <td>0.205690</td>
      <td>0.127006</td>
      <td>0.556386</td>
      <td>0.360075</td>
      <td>0.508442</td>
      <td>0.374508</td>
      <td>0.483590</td>
      <td>0.385375</td>
      <td>0.359744</td>
      <td>0.835052</td>
      <td>0.403706</td>
      <td>0.213433</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.210090</td>
      <td>0.360839</td>
      <td>0.233501</td>
      <td>0.102906</td>
      <td>0.811321</td>
      <td>0.811361</td>
      <td>0.565604</td>
      <td>0.522863</td>
      <td>0.776263</td>
      <td>1.000000</td>
      <td>0.139091</td>
      <td>0.175875</td>
      <td>0.126655</td>
      <td>0.038155</td>
      <td>0.251453</td>
      <td>0.543215</td>
      <td>0.142955</td>
      <td>0.353665</td>
      <td>0.728148</td>
      <td>0.287205</td>
      <td>0.248310</td>
      <td>0.385928</td>
      <td>0.241347</td>
      <td>0.094008</td>
      <td>0.915472</td>
      <td>0.814012</td>
      <td>0.548642</td>
      <td>0.884880</td>
      <td>1.000000</td>
      <td>0.773711</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.629893</td>
      <td>0.156578</td>
      <td>0.630986</td>
      <td>0.489290</td>
      <td>0.430351</td>
      <td>0.347893</td>
      <td>0.463918</td>
      <td>0.518390</td>
      <td>0.378283</td>
      <td>0.186816</td>
      <td>0.233822</td>
      <td>0.093065</td>
      <td>0.220563</td>
      <td>0.163688</td>
      <td>0.332359</td>
      <td>0.167918</td>
      <td>0.143636</td>
      <td>0.357075</td>
      <td>0.136179</td>
      <td>0.145800</td>
      <td>0.519744</td>
      <td>0.123934</td>
      <td>0.506948</td>
      <td>0.341575</td>
      <td>0.437364</td>
      <td>0.172415</td>
      <td>0.319489</td>
      <td>0.558419</td>
      <td>0.157500</td>
      <td>0.142595</td>
    </tr>
  </tbody>
</table>
</div>




```python
#dividing data to have a training and a testing set
X_train, X_test, y_train, y_test = train_test_split(Xdf, y, test_size= .4, random_state=0)
```


```python
# setting Logistic regression classifier
# scikit-learn includes the intercept.

lr = LogisticRegression(C=1e9)

#dividing data to have a training and a testing set
X_train, X_test, y_train, y_test = train_test_split(Xdf, y, test_size= .4, random_state=0)

# Logistic regression cross validation
Kfold = KFold(len(ranked_predictors), shuffle=False)
print("KfoldCrossVal mean score using Logistic regression is %s" %cross_val_score(lr,Xdf,y,cv=10).mean())

# Logistic regression metrics
LRm = lr.fit(X_train, y_train)

LRm.predict_proba(X_test)  # The returned estimates for all classes are ordered by the label of classes.

```

    KfoldCrossVal mean score using Logistic regression is 0.9508458646616542
    




    array([[0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 9.44613526e-162],
           [1.00000000e+000, 1.48397981e-277],
           [1.00000000e+000, 4.45065541e-181],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 6.22216697e-232],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 6.08211414e-172],
           [1.00000000e+000, 2.29149077e-158],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 3.06885187e-100],
           [1.00000000e+000, 3.90861983e-044],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 2.84276261e-260],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 1.01942789e-158],
           [1.00000000e+000, 8.63001772e-288],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 1.54275638e-304],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 2.05159668e-026],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 5.23373576e-036],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 8.19445570e-233],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 3.83837796e-298],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 4.74342961e-093],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 5.17720057e-237],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [3.55271368e-015, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 2.40200568e-221],
           [1.00000000e+000, 6.76865471e-294],
           [1.00000000e+000, 3.14532978e-260],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 1.76634298e-296],
           [1.00000000e+000, 2.08807151e-183],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 3.70456312e-067],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 3.66547221e-223],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 1.09442458e-213],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 6.21883634e-222],
           [1.00000000e+000, 2.12515256e-241],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 1.91651717e-259],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [5.65419120e-005, 9.99943458e-001],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 4.90345333e-303],
           [1.00000000e+000, 2.03846246e-273],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 8.99360068e-153],
           [1.00000000e+000, 3.24894474e-219],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 4.07737870e-261],
           [1.00000000e+000, 3.63056502e-293],
           [1.00000000e+000, 2.62692618e-104],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 7.34871035e-131],
           [1.00000000e+000, 1.42893536e-020],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 1.25042036e-204],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 4.92444347e-111],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 2.50372633e-139],
           [1.00000000e+000, 1.84093675e-072],
           [1.00000000e+000, 2.14998070e-276],
           [1.00000000e+000, 1.53087980e-134],
           [1.00000000e+000, 0.00000000e+000],
           [3.04307202e-001, 6.95692798e-001],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 8.25351280e-103],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 1.03600959e-053],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 8.17742570e-119],
           [1.00000000e+000, 2.16519374e-151],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 1.13630060e-256],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 1.04304737e-250],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 4.31926783e-202],
           [1.28592479e-003, 9.98714075e-001],
           [1.00000000e+000, 5.36702792e-284],
           [1.00000000e+000, 2.64697067e-281],
           [1.00000000e+000, 1.46164822e-261],
           [1.00000000e+000, 2.67157156e-229],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 6.24029766e-019],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 2.01978424e-114],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 3.36718226e-091],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 3.08731311e-048],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 3.67520404e-308],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 1.08087654e-193],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 4.88593509e-182],
           [4.60170355e-003, 9.95398296e-001],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 3.16406816e-251],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 6.79392857e-135],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 3.18772632e-155],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 1.10195214e-103],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 9.19748572e-056],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 1.48909835e-100],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 5.15441267e-153],
           [0.00000000e+000, 1.00000000e+000],
           [1.00000000e+000, 3.33226303e-128],
           [1.00000000e+000, 3.64558102e-179],
           [1.00000000e+000, 0.00000000e+000],
           [1.00000000e+000, 0.00000000e+000],
           [0.00000000e+000, 1.00000000e+000]])




```python
LRm.decision_function(X_test) # Predict confidence scores for samples.
```




    array([ 1.24377648e+03, -3.70773179e+02, -6.37421343e+02, -4.15274850e+02,
           -8.47699440e+02, -5.32371623e+02, -8.96976468e+02, -7.27643334e+02,
           -7.20512826e+02, -1.61335324e+03, -3.94239284e+02, -3.62979242e+02,
           -1.09768464e+03, -2.29137206e+02, -9.99505598e+01,  1.03605287e+03,
           -5.97627348e+02,  1.69058320e+03,  1.01798928e+03,  2.72683759e+03,
            1.21572615e+03,  3.67091196e+02, -3.63789203e+02, -6.60989260e+02,
            6.41846245e+02, -7.11411528e+02, -8.01476173e+02,  1.32895829e+02,
           -6.99552298e+02,  1.66781996e+03, -9.16253747e+02,  1.57008134e+03,
           -5.91485941e+01,  4.79097415e+02, -1.29580931e+03,  1.05215916e+03,
           -8.12379380e+01,  9.62953962e+02, -5.34398869e+02,  1.47126013e+03,
            4.26989650e+02, -6.84825308e+02,  5.06188397e+02, -1.05266275e+03,
           -2.12583653e+02,  3.04304850e+03, -1.33459717e+03, -5.44068403e+02,
           -9.26562917e+02,  1.05877306e+03,  9.51589038e+02,  3.32958747e+01,
            1.44925322e+03, -5.07995001e+02, -6.75047715e+02, -5.97526205e+02,
           -8.84910327e+02, -6.80996276e+02, -4.20636831e+02,  2.64841261e+03,
            1.05639903e+03,  1.30923069e+03, -9.11230197e+02, -7.81295409e+02,
            1.28034748e+03, -1.52963636e+02,  3.50590574e+03,  1.82842530e+03,
            1.84542071e+03, -7.15721742e+02,  7.74403731e+01,  1.60022966e+03,
           -5.12177519e+02,  4.33715503e+02,  1.80305181e+03, -4.90360396e+02,
           -7.69122253e+02, -5.09346308e+02, -5.54169164e+02, -1.34013562e+03,
            5.40225412e+02,  2.32019092e+03,  1.66605126e+03, -9.23771648e+02,
            5.21246949e+02, -5.95719030e+02, -1.01111465e+03, -1.20961334e+03,
            1.46811162e+03,  2.41187259e+03, -8.38162112e+02,  2.50299037e+02,
            9.78047185e+00,  1.58961987e+03, -6.96093343e+02, -6.27893535e+02,
            3.17553243e+03, -3.50099006e+02, -5.03087805e+02, -8.61779745e+02,
           -2.49752664e+03, -5.99569255e+02, -6.73368044e+02, -2.38503035e+02,
            1.73392325e+03, -1.08276006e+03,  1.24729435e+03, -2.99644122e+02,
           -4.56947722e+01, -7.60708963e+02, -4.69503879e+02,  1.39717135e+03,
            1.20844799e+03, -2.53992734e+02, -8.10874350e+02, -3.19141548e+02,
           -1.65175852e+02, -6.34748027e+02, -3.08120560e+02, -7.58675109e+02,
            8.26870457e-01, -7.66649616e+02, -1.06751925e+03, -2.35055626e+02,
            2.15878686e+03, -1.22001634e+02,  1.87775979e+02, -7.40521138e+02,
           -1.05288331e+03, -1.06187933e+03,  1.22092746e+03, -8.77173708e+02,
           -1.00941346e+03, -2.71906249e+02, -3.46917839e+02, -1.19108750e+03,
            8.68726473e+01,  2.44211331e+02,  3.85607200e+03, -1.22728896e+03,
           -5.89334006e+02, -1.09614495e+03,  2.73141083e+03, -5.75604127e+02,
           -1.02746853e+03,  1.82238040e+03, -4.63659103e+02,  6.65499039e+00,
           -6.52253892e+02, -6.46052995e+02, -6.00595145e+02, -5.26309319e+02,
            4.97455993e+02, -4.19180889e+01, -9.92117511e+02,  6.17526128e+02,
           -2.61791710e+02,  5.99142106e+02, -7.88145368e+02,  1.92546911e+03,
            1.81243272e+03, -8.18939211e+02, -2.08321167e+02,  8.64636991e+02,
           -8.15506289e+02,  1.25897984e+03,  5.87401500e+02,  9.11634421e+02,
           -1.09396783e+02, -7.20035794e+02, -9.27377888e+02, -7.07894600e+02,
           -8.94335420e+02, -4.44321151e+02,  7.23759977e+02, -1.41917762e+03,
           -9.90389759e+02, -9.02180906e+02, -8.42240591e+02,  6.89380597e+02,
            2.13442721e+03, -4.17484126e+02,  5.37671638e+00,  7.22275899e+02,
           -1.13482120e+03, -7.43186155e+02,  1.90309498e+02, -5.76797000e+02,
            2.28126629e+03, -1.08785213e+03, -8.99524884e+02,  1.52147687e+03,
            6.91628909e+02, -9.98827395e+02, -9.31098236e+02,  2.26539191e+03,
            2.23336151e+03, -7.97846024e+02, -3.08932958e+02,  6.93915365e+02,
           -3.55741382e+02, -9.61055291e+02,  1.76406388e+02,  3.55263248e+03,
            1.46908926e+03, -1.24176496e+03, -2.37069181e+02, -1.19246723e+03,
            1.70521388e+03, -8.65272016e+02, -1.26725835e+02, -1.05038922e+03,
           -8.33637070e+02, -1.02102856e+03,  3.62463044e+03, -1.01760070e+03,
            2.94862504e+03, -2.29860338e+02,  6.73298813e+02, -7.36873338e+02,
            6.44124352e+02, -3.50655666e+02,  8.34159982e+02, -2.93527240e+02,
           -4.10869216e+02, -8.00602681e+02, -7.13076920e+02,  9.12947535e+02])




```python
LRm.predict(X_test)
```




    array(['M', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
           'B', 'B', 'M', 'B', 'M', 'M', 'M', 'M', 'M', 'B', 'B', 'M', 'B',
           'B', 'M', 'B', 'M', 'B', 'M', 'B', 'M', 'B', 'M', 'B', 'M', 'B',
           'M', 'M', 'B', 'M', 'B', 'B', 'M', 'B', 'B', 'B', 'M', 'M', 'M',
           'M', 'B', 'B', 'B', 'B', 'B', 'B', 'M', 'M', 'M', 'B', 'B', 'M',
           'B', 'M', 'M', 'M', 'B', 'M', 'M', 'B', 'M', 'M', 'B', 'B', 'B',
           'B', 'B', 'M', 'M', 'M', 'B', 'M', 'B', 'B', 'B', 'M', 'M', 'B',
           'M', 'M', 'M', 'B', 'B', 'M', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
           'M', 'B', 'M', 'B', 'B', 'B', 'B', 'M', 'M', 'B', 'B', 'B', 'B',
           'B', 'B', 'B', 'M', 'B', 'B', 'B', 'M', 'B', 'M', 'B', 'B', 'B',
           'M', 'B', 'B', 'B', 'B', 'B', 'M', 'M', 'M', 'B', 'B', 'B', 'M',
           'B', 'B', 'M', 'B', 'M', 'B', 'B', 'B', 'B', 'M', 'B', 'B', 'M',
           'B', 'M', 'B', 'M', 'M', 'B', 'B', 'M', 'B', 'M', 'M', 'M', 'B',
           'B', 'B', 'B', 'B', 'B', 'M', 'B', 'B', 'B', 'B', 'M', 'M', 'B',
           'M', 'M', 'B', 'B', 'M', 'B', 'M', 'B', 'B', 'M', 'M', 'B', 'B',
           'M', 'M', 'B', 'B', 'M', 'B', 'B', 'M', 'M', 'M', 'B', 'B', 'B',
           'M', 'B', 'B', 'B', 'B', 'B', 'M', 'B', 'M', 'B', 'M', 'B', 'M',
           'B', 'M', 'B', 'B', 'B', 'B', 'M'], dtype=object)




```python
y_pred = LRm.predict(X_test)
print("Accuracy score using Logistic regression is %s" %metrics.accuracy_score(y_test, y_pred))
```

    Accuracy score using Logistic regression is 0.9517543859649122
    

## Useful Links

ROC Curves and Area Under the Curve (AUC) Explained https://youtu.be/OAl6eAyP-yo

This video clearly explains how to decide a classification threshold.


