# I2SL Linear Regression


## Linear Models

Linear regression predicts the response variable $y$ assuming it has a linear relationship with predictor variable(s) $x$ or $x_1, x_2, ,,, x_n$.

$$y = \beta_0 + \beta_1 x + \varepsilon .$$

*Simple* regression use only one predictor variable $x$. *Mulitple* regression uses a set of predictor variables $x_1, x_2, ,,, x_n$.

The *response variable* $y$ is also called the regressand, forecast, dependent or explained variable. The *predictor variable* $x$ is also called the regressor, independent or explanatory variable.

The parameters $\beta_0$ and $\beta_1$ determine the intercept and the slope of the line respectively. The intercept $\beta_0$ represents the predicted value of $y$ when $x=0$. The slope $\beta_1$ represents the predicted increase in $Y$ resulting from a one unit increase in $x$.

Note that the regression equation is just our famliar equation for a line with an error term.

The equation for a line:  
$$ Y = bX + a $$

$$y = \beta_0 + \beta_1 x $$

The equation for a line with an error term:  

$$ Y = bX + a + \varepsilon $$

$$y = \beta_0 + \beta_1 x + \varepsilon .$$

- $b$ = $\beta_1$ = slope
- $a$ = $\beta_0$ = $Y$ intercept
- $\varepsilon$ = error term


We can think of each observation $y_i$ consisting of the systematic or explained part of the model, $\beta_0+\beta_1x_i$, and the random *error*, $\varepsilon_i$.

_Zero Slope_

Note that when  $\beta_1 = 0$ then response does not change as the predictor changes.

For multiple regression $x$ is a $X$ to produce a system of equations:  

$$ Y = \beta_0 + \beta_1 X  + \varepsilon $$

## The error $\varepsilon_i$

The error term is a catch-all for anything that may affect $y_i$ other than $x_i$. We assume that these errors:

* have mean zero; otherwise the forecasts will be systematically biased.
* statistical independence of the errors (in particular, no correlation between consecutive errors in the case of time series data).
* homoscedasticity (constant variance) of the errors.
* normality of the error distribution.

If any of these assumptions is violated then the robustness of the model to be taken with a grain of salt.


## Least squares estimation

In a linear model, the values of $\beta_0$ and $\beta_1$. These need to be estimated from the data. We call this *fitting a model*.

The least squares method iis the most common way of estimating $\beta_0$ and $\beta_1$ by minimizing the sum of the squared errors. The values of $\beta_0$ and $\beta_1$ are chosen so that that minimize

$$\sum_{i=1}^N \varepsilon_i^2 = \sum_{i=1}^N (y_i - \beta_0 - \beta_1x_i)^2. $$


Using mathematical calculus, it can be shown that the resulting **least squares estimators** are

$$\hat{\beta}_1=\frac{ \sum_{i=1}^{N}(y_i-\bar{y})(x_i-\bar{x})}{\sum_{i=1}^{N}(x_i-\bar{x})^2} $$

and

$$\hat{\beta}_0=\bar{y}-\hat{\beta}_1\bar{x}, $$

where $\bar{x}$ is the average of the $x$ observations and $\bar{y}$ is the average of the $y$ observations. The estimated line is known as the *regression line*.

To solve least squares with gradient descent or stochastic gradient descent (SGD) or losed Form (set derivatives equal to zero and solve for parameters).

## Fitted values and residuals

The response values of $y$ obtained from the observed $x$ values are
called *fitted values*: $\hat{y}_i=\hat{\beta}_0+\hat{\beta}_1x_i$, for
$i=1,\dots,N$. Each $\hat{y}_i$ is the point on the regression
line corresponding to $x_i$.

The difference between the observed $y$ values and the corresponding fitted values are the *residuals*:

$$e_i = y_i - \hat{y}_i = y_i -\hat{\beta}_0-\hat{\beta}_1x_i. $$

The residuals have some useful properties including the following two:

$$\sum_{i=1}^{N}{e_i}=0 \quad\text{and}\quad \sum_{i=1}^{N}{x_ie_i}=0. $$

![Linear regression](images/Linear_regression.svg.png)

Residuals are the errors that we cannot predict.Residuals are highly useful for studying whether a given regression model is an appropriate statistical technique for analyzing the relationship.

## Linear regression and correlation

The correlation coefficient $r$ measures the strength and the direction of the linear relationship between the two variables. The stronger the linear relationship, the closer the observed data points will cluster around a straight line.

The _Pearson product-moment correlation coefficient_ is the most widely used of all correlation coefficients. In statistics, the Pearson product-moment correlation coefficient (/ˈpɪərsɨn/) (sometimes referred to as the PPMCC or PCC or Pearson's r) is a measure of the linear correlation (dependence) between two variables X and Y, giving a value between +1 and −1 inclusive, where 1 is total positive correlation, 0 is no correlation, and −1 is total negative correlation. It is widely used in the sciences as a measure of the degree of linear dependence between two variables. It was developed by Karl Pearson from a related idea introduced by Francis Galton in the 1880s. Early work on the distribution of the sample correlation coefficient was carried out by Anil Kumar Gain and R. A. Fisher from the University of Cambridge.

from [Pearson product-moment correlation coefficient](https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient)  

Examples of scatter diagrams with different values of correlation coefficient (ρ)

The value of r is such that -1 < r < +1.  

Strong positive correlation r is close to +1. Strong negative correlation r is close to -1. No correlation r is close to 0.   

![Correlation_examples](images/Correlation_examples.png)

The advantage of a regression model over correlation is that it asserts a predictive relationship between the two variables ($x$ predicts $y$) and quantifies this in a useful way for forecasting.


We can have a _positive linear relationship_ (r>0), _negative linear relationship_ (r<0), or _no linear relationship_ (r=0) (Note that no linear relationship doesn't mean no relationship.)

![Anscombe's quartet](images/Anscombes_quartet.svg)


Anscombe's quartet comprises four datasets that have nearly identical simple statistical properties, yet appear very different when graphed. They were constructed in 1973 by the statistician Francis Anscombe to demonstrate both the importance of graphing data before analyzing it and the effect of outliers on statistical properties.

from [Anscombe's quartet - Wikipedia](https://en.wikipedia.org/wiki/Anscombe%27s_quartet)


## The Modeling process  

A. Formula (i.e. what $f(x)$ and which $x_1,x_2, .. x_n$)   
B. Fit (i.e. Estimate the unknown parameters for the model.)   
C. Analysis of fit (i.e. how good is the model)   
D. Analysis of residuals (i.e. how closely did the model match assumptions)   

We often create many models so we *_store & explore_*. That is, make models and save them as variables so we can compare the various iterations of the modeling process.


## A. Formula

f(x)          | x    
------------- | -------------  
response      | ~ predictor     
response      | ~ explanatory   
dependent     | ~ independent    
outcome       | ~ predictor  
forecast      | ~ predictor   
regressand    | ~ regressor  
explained     | ~ explanatory  






```python
import pandas as pd
import pandas.util.testing as tm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
df_adv = pd.read_csv('https://raw.githubusercontent.com/nikbearbrown/Google_Colab/master/data/advertising.csv')
# Sort Sales of individual department for plotting
df_adv_new = pd.melt(df_adv, value_vars=['TV', 'Radio', 'Newspaper'], id_vars='Sales', value_name='adv_budget')
#linear model
lm = sns.lmplot('adv_budget', 'Sales', data=df_adv_new, col='variable', sharey=False, sharex=False, lowess=True);
axes = lm.axes
for i, ax in enumerate(axes[0]):
    ax.set_xlim(0,)
    ax.set_title(lm.col_names[i])
    ax.set_xlabel('Advertising Budget')
```


    
![png](output_2_0.png)
    


## Creating a linear model

Consider the advertising data shown above.

Questions we might ask:
- Is there a relationship between advertising budget and sales?
- How strong is the relationship between advertising budget and sales?
- Which media contribute to sales?
- How accurately can we predict future sales?
- Is the relationship linear?
- Is there synergy among the advertising media?

### Simple linear regression using a single predictor X

- We assume a model
\begin{equation}y=\beta_{0}+\beta_{1} X+\epsilon\end{equation}
where $\beta_{0}$ and $\beta_{1}$ are two unknown constants that represent
the intercept and slope, also known as coefficients or
parameters, and $\epsilon$ is the error term.
- Given some estimates $\hat{\beta}_{0}$ and $\hat{\beta}_{1}$ for the model coefficients,
we predict future sales using
\begin{equation}\hat{y}=\hat{\beta}_{0}+\hat{\beta}_{1} x\end{equation}
where $\hat{y}$ indicates a prediction of Y on the basis of X = x.
The hat symbol denotes an estimated value.


### Estimation of the slope and intercept parameters by least squares   

- Let $\hat{y}_{i}=\hat{\beta}_{0}+\hat{\beta}_{1} x_{i}$ be the prediction for Y based on the ith value of X. Then $e_{i}=y_{i}-\hat{y}_{i}$ represents the ith residual
- We define the residual sum of squares (RSS) as
$\mathrm{RSS}=e_{1}^{2}+e_{2}^{2}+\cdots+e_{n}^{2}$
or equivalently as
\begin{equation}\mathrm{RSS}=\left(y_{1}-\beta_{0}-\beta_{1} x_{1}\right)^{2}+\left(y_{2}-\hat{\beta}_{0}-\beta_{1} x_{2}\right)^{2}+\ldots+\left(y_{n}-\hat{\beta}_{0}-\hat{\beta}_{1} x_{n}\right)^{2}\end{equation}

- The least squares approach chooses βˆ
0 and βˆ
1 to minimize
the RSS. The minimizing values can be shown to be
\begin{equation}\hat{\beta}_{1}=\frac{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right)}{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}}\end{equation},
\begin{equation}\hat{\beta}_{0}=\bar{y}-\hat{\beta}_{1} \bar{x}\end{equation}
where $\bar{y} \equiv \frac{1}{n} \sum_{i=1}^{n} y_{i}$ and $\bar{x} \equiv \frac{1}{n} \sum_{i=1}^{n} x_{i}$ are the sample
means


## Fitting a degree one polynomial

Let's use np.polyfit to fit a degree one polynomial (i.e. a line)



```python
# Fitting a degree one polynomial
fit = np.polyfit(df_adv['Newspaper'], df_adv['Sales'], deg=1)
y_hat = fit[1] + df_adv['Newspaper'] * fit[0]

plt.figure(figsize=(8, 6))
sns.regplot('Newspaper', 'Sales', data=df_adv)
plt.vlines(df_adv['Newspaper'], y_hat, df_adv['Sales'], lw = .4);

```


    
![png](output_4_0.png)
    



```python
fit = np.polyfit(df_adv['Radio'], df_adv['Sales'], deg=1)
y_hat = fit[1] + df_adv['Radio'] * fit[0]

plt.figure(figsize=(8, 6))
sns.regplot('Radio', 'Sales', data=df_adv)
plt.vlines(df_adv['Radio'], y_hat, df_adv['Sales'], lw = .4);
```


    
![png](output_5_0.png)
    



```python

fit = np.polyfit(df_adv['TV'], df_adv['Sales'], deg=1)
y_hat = fit[1] + df_adv['TV'] * fit[0]

plt.figure(figsize=(8, 6))
sns.regplot('TV', 'Sales', data=df_adv)
plt.vlines(df_adv['TV'], y_hat, df_adv['Sales'], lw = .4);
```


    
![png](output_6_0.png)
    


The least squares fit for the regression of sales onto TV.
In this case a linear fit captures the essence of the relationship,
although it is somewhat deficient in the left of the plot.

## Discussion

How do we know if the fit is good?


## Assessing the Accuracy of the Coefficient Estimates

- The standard error of an estimator reflects how it varies under repeated sampling. We have
\begin{equation}\operatorname{SE}\left(\hat{\beta}_{1}\right)^{2}=\frac{\sigma^{2}}{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}}, \quad \operatorname{SE}\left(\hat{\beta}_{0}\right)^{2}=\sigma^{2}\left[\frac{1}{n}+\frac{\bar{x}^{2}}{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}}\right]\end{equation}
where $\sigma^{2}=\operatorname{Var}(\epsilon)$

- These standard errors can be used to compute confidence intervals. A 95% **confidence interval** is defined as a range of values such that with 95% probability, the range will contain the true unknown value of the parameter. It has the form
\begin{equation}\hat{\beta}_{1} \pm 2 \cdot \operatorname{SE}\left(\hat{\beta}_{1}\right)\end{equation}

That is, there is approximately a 95% chance that the interval
\begin{equation}\left[\hat{\beta}_{1}-2 \cdot \operatorname{SE}\left(\hat{\beta}_{1}\right), \hat{\beta}_{1}+2 \cdot \operatorname{SE}\left(\hat{\beta}_{1}\right)\right]\end{equation}
will contain the true value of β1 (under a scenario where we got repeated samples like the present sample)
For the advertising data, the 95% confidence interval for β1 is
[0.042, 0.053]

- Standard errors can also be used to perform $\textit{hypothesis tests}$ on the coefficients. The most common hypothesis test involves testing the $\textit{null hypothesis}$ of

    $H_{0}$ : There is no relationship between X and Y versus the $\textit{alternative hypothesis}$

    $H_{A}$ : There is some relationship between X and Y .

- Mathematically, this corresponds to testing
\begin{equation}H_{0}: \beta_{1}=0\end{equation}
versus
\begin{equation}H_{A}: \beta_{1} \neq 0\end{equation}
since if $\beta_1=0$ then the model reduces to Y = $\beta_{0}$ + $\epsilon$, and X is not associated with Y .

- To test the null hypothesis, we compute a t-statistic, given by
\begin{equation}t=\frac{\hat{\beta}_{1}-0}{\operatorname{SE}\left(\hat{\beta}_{1}\right)}\end{equation}
- This will have a t-distribution with n − 2 degrees of freedom, assuming $\beta_{1}$ = 0.
- Using statistical software, it is easy to compute the probability of observing any value equal to |t| or larger. We call this probability the p-value


## Building a multivariate model with statsmodels

Let's fit all three independent variables (TV, Radio, and Newspaper) to fit a  multivariate model to predict sales.

Note that the package uses an "R-like" syntax to specify the dependent variable ~ independent variables relationship.

```python
Sales ~ TV + Newspaper + Radio
```




```python
import statsmodels.formula.api as smf
results = smf.ols('Sales ~ TV + Newspaper + Radio', data=df_adv).fit()
results.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>Sales</td>      <th>  R-squared:         </th> <td>   0.897</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.896</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   570.3</td>
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 14 Sep 2020</td> <th>  Prob (F-statistic):</th> <td>1.58e-96</td>
</tr>
<tr>
  <th>Time:</th>                 <td>01:21:36</td>     <th>  Log-Likelihood:    </th> <td> -386.18</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   200</td>      <th>  AIC:               </th> <td>   780.4</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   196</td>      <th>  BIC:               </th> <td>   793.6</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    2.9389</td> <td>    0.312</td> <td>    9.422</td> <td> 0.000</td> <td>    2.324</td> <td>    3.554</td>
</tr>
<tr>
  <th>TV</th>        <td>    0.0458</td> <td>    0.001</td> <td>   32.809</td> <td> 0.000</td> <td>    0.043</td> <td>    0.049</td>
</tr>
<tr>
  <th>Newspaper</th> <td>   -0.0010</td> <td>    0.006</td> <td>   -0.177</td> <td> 0.860</td> <td>   -0.013</td> <td>    0.011</td>
</tr>
<tr>
  <th>Radio</th>     <td>    0.1885</td> <td>    0.009</td> <td>   21.893</td> <td> 0.000</td> <td>    0.172</td> <td>    0.206</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>60.414</td> <th>  Durbin-Watson:     </th> <td>   2.084</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 151.241</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-1.327</td> <th>  Prob(JB):          </th> <td>1.44e-33</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.332</td> <th>  Cond. No.          </th> <td>    454.</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



## So what do all of the stats in that table mean?  

_The coefficients_

There are three independent variables

$$ f(x) = \beta_0 + \beta_1 x + \beta_2 x + \beta_3 x + \varepsilon . $$

or more specifically

$$ Sales = \beta_0 + \beta_1 TV + \beta_2 Radio + \beta_3 Newspaper + \varepsilon . $$

A slope like $\beta_1$ indicates for each one unit increase in TV we expect a $\beta_1$ increase or decrease in sales (remember that lines can be negative)

The y-intercept  $\beta_0$ indicates our expected sales if we didn't spend anything on TV, Radio or Newspaper. That is

$$ Sales = \beta_0 + \beta_1 *0 + \beta_2 *0 + \beta_3 *0 + \varepsilon$$

Note that the error $\varepsilon$ is not a function of the independent variables.

Choosing predictors (independent variables) is typically done in 3 ways:  

1. Theory / domain knowledge
2. EDA - exploritory data analysis  
3. Fitting various combinations of outcome and predictors and selecting the best fits (r.g. Stepwise regression)  
4. Feature selection  






```python
print('Parameters: ', results.params)
print('R2: ', results.rsquared)
```

    Parameters:  Intercept    2.938889
    TV           0.045765
    Newspaper   -0.001037
    Radio        0.188530
    dtype: float64
    R2:  0.8972106381789522
    

#### Multiple Linear Regression

- Here our model is

\begin{equation}Y=\beta_{0}+\beta_{1} X_{1}+\beta_{2} X_{2}+\cdots+\beta_{p} X_{p}+\epsilon\end{equation}

- We interpret βj as the average effect on Y of a one unit increase in Xj , holding all other predictors fixed. In the advertising example, the model becomes

\begin{equation}\text {Sales}=\beta_{0}+\beta_{1} \times \mathrm{TV}+\beta_{2} \times \mathrm{Radio}+\beta_{3} \times \text { Newspaper }+\epsilon\end{equation}


therefore

$$ Sales = 2.9 + 0.046  \mathrm{TV} +  0.19 \mathrm{Radio} + -0.001 \mathrm{Newspaper} + \varepsilon$$

Note Bien: The model tells one that newspaper advertising actually hurts sales due to its negative slope.  

Note Bien: The units of the independent variables are not standardized as they may have different units. One probably wants to know the dollar increase in sales for each dollar spent.  


### Interpreting regression coefficients

- The ideal scenario is when the predictors are uncorrelated a balanced design:
    - Each coefficient can be estimated and tested separately.
    - Interpretations such as “a unit change in Xj is associated with a βj change in Y , while all the other variables stay fixed”, are possible.
- Correlations amongst predictors cause problems:
    - The variance of all coefficients tends to increase, sometimes dramatically
    - Interpretations become hazardous — when Xj changes, everything else changes.
- Claims of causality should be avoided for observational data.


### The woes of (interpreting) regression coefficients

“Data Analysis and Regression” Mosteller and Tukey 1977
- a regression coefficient βj estimates the expected change in Y per unit change in Xj , with all other predictors held fixed. But predictors usually change together!
- Example: Y total amount of change in your pocket; X1 = # of coins; X2 = # of pennies, nickels and dimes. By itself, regression coefficient of Y on X2 will be > 0. But how about with X1 in model?
- Y = number of tackles by a football player in a season; W and H are his weight and height. Fitted regression model is $\hat{Y}=b_{0}+.50 W-.10 H$. How do we interpret βˆ 2 < 0?

### Two quotes by famous Statisticians

“Essentially, all models are wrong, but some are useful”
George Box

“The only way to find out what will happen when a complex system is disturbed is to disturb the system, not merely to observe it passively”
Fred Mosteller and John Tukey, paraphrasing George Box

### Estimation and Prediction for Multiple Regression

- Given estimates $\hat{\beta}_{0}, \hat{\beta}_{1}, . . . \hat{\beta}_{p}$, we can make predictions using the formula
\begin{equation}\hat{y} = \hat{\beta}_{0} + \hat{\beta}_{0} x_{1} + \hat{\beta}_{0} x_{2} + · · · + \hat{\beta}_{0} x_{p}\end{equation}
- We estimate β0, β1, . . . , βp as the values that minimize the sum of squared residuals

This is done using standard statistical software. The values $\hat{\beta}_{0}, \hat{\beta}_{1}, . . . \hat{\beta}_{p}$ that minimize RSS are the multiple least squares regression coefficient estimates.
\begin{equation}\begin{aligned}
\mathrm{RSS} &=\sum_{i=1}^{n}\left(y_{i}-\hat{y}_{i}\right)^{2} \\
&=\sum_{i=1}^{n}\left(y_{i}-\hat{\beta}_{0}-\hat{\beta}_{1} x_{i 1}-\hat{\beta}_{2} x_{i 2}-\cdots-\hat{\beta}_{p} x_{i p}\right)^{2}
\end{aligned}\end{equation}   


### = Some important questions

1. Is at least one of the predictors X1, X2, . . . , Xp useful in predicting the response?
2. Do all the predictors help to explain Y , or is only a subset of the predictors useful?
3. How well does the model fit the data?
4. Given a set of predictor values, what response value should we predict, and how accurate is our prediction?

### Is atleast one predictor useful?

For the first question, we can use the F-statistic
\begin{equation}F=\frac{(\mathrm{TSS}-\mathrm{RSS}) / p}{\mathrm{RSS} /(n-p-1)} \sim F_{p, n-p-1}\end{equation}  


Deciding on the important variables

- The most direct approach is called all subsets or best subsets regression: we compute the least squares fit for all possible subsets and then choose between them based on some criterion that balances training error with model size.
- However we often can’t examine all possible models, since they are $2^{p}$ of them; for example when p = 40 there are over a billion models! Instead we need an automated approach that searches through a subset of them. We discuss two commonly use approaches next.

### Forward selection
- Begin with the null model — a model that contains an intercept but no predictors.
- Fit p simple linear regressions and add to the null model the variable that results in the lowest RSS.
- Add to that model the variable that results in the lowest RSS amongst all two-variable models.
- Continue until some stopping rule is satisfied, for example when all remaining variables have a p-value above some threshold.

### Backward selection
- Start with all variables in the model.
- Remove the variable with the largest p-value — that is, the variable that is the least statistically significant.
- The new (p − 1)-variable model is fit, and the variable with the largest p-value is removed.
- Continue until a stopping rule is reached. For instance, we may stop when all remaining variables have a significant p-value defined by some significance threshold.

### Model selection — continued
- Later we discuss more systematic criteria for choosing an “optimal” member in the path of models produced by forward or backward stepwise selection.
- These include Mallow’s Cp, Akaike information criterion (AIC), Bayesian information criterion (BIC), adjusted R2 and Cross-validation (CV).

### Qualitative Predictors
- Some predictors are not quantitative but are qualitative, taking a discrete set of values.
- These are also called categorical predictors or factor variables.
- See for example the scatterplot matrix of the credit card data in the next slide. In addition to the 7 quantitative variables shown, there are four qualitative variables: gender, student (student status), status (marital status), and ethnicity (Caucasian, African American (AA) or Asian).


## Assessing the model fit  

To evaluate a regression model we ask the following questions:

A. Does it make sense?  
B. Is the "true" $\beta_1$ significantly differnet from  $\beta_1 = 0$?  
C. Are any assumptions of the model violated?  
D. How tightly the parameter estimation fits the residuals?  


## Hypothesis testing: Is he "true" $\beta_1 \neq 0$?

Recall that when the slope $\beta_1 = 0$ we have no relationship between the outcome and predictors.

Hypothesis tests assume the thing you want to disprove, and then to look for evidence that the assumption is wrong. In this case, we assume that there is no relationship between $x$ and $f(x)$. This is called the *null hypothesis* and is stated as

$$H_0: \beta_1 = 0$$

Evidence against this hypothesis is provided by the value of $\hat{\beta}_1$, the slope estimated from the data. If $\hat{\beta}_1$ is very different from zero, we conclude that the null hypothesis is incorrect and that the evidence suggests there really is a relationship between $x$ and $f(x)$.

There are many hypothesis tests that can be used to test whether the "true" $\beta_1 \neq 0$:

* Student’s T-Tests
* One-Sample T-Test
* Two-Sample T-Test
* Paired T-Test
* Wilcoxon Rank-Sum Test
* Analysis of Variance (ANOVA)
* Kruskal-Wallis Test

We will discuss these more in the module on hypothesis testing. As R's lm() function gives p-values by default we will focus on them.

## P-value

To determine how big the difference between $\hat{\beta}_1$ (the "true"  $\beta_1$) and $\beta_1$ must be before we would reject the null hypothesis, we calculate the probability of obtaining a value of $\beta_1$ as large as we have calculated if the null hypothesis were true. This probability is known as the *P-value*.

In statistics, the p-value is a function of the observed sample results (a statistic) that is used for testing a statistical hypothesis. Before the test is performed, a threshold value is chosen, called the significance level of the test, traditionally 5% or 1% and denoted as $\alpha$.  

If the p-value is equal to or smaller than the significance level ($\alpha$), it suggests that the observed data are inconsistent with the assumption that the null hypothesis is true and thus that hypothesis must be rejected (but this does not automatically mean the alternative hypothesis can be accepted as true). When the p-value is calculated correctly, such a test is guaranteed to control the Type I error rate to be no greater than $\alpha$.

from [P-value](https://en.wikipedia.org/wiki/P-value)

## Confidence intervals

In statistics, a confidence interval (CI) is a type of interval estimate of a population parameter. It provides an interval estimate for lower or upper confidence bounds. For $\beta_1$, usually referred to as a *confidence interval* and is typically +/-0.5% (a 99% confidence interval),+/-1% (a 98% confidence interval),+/-2.5% (a 95% confidence interval) or +/-5% (a 90% confidence interval). The lower and upper confidence bounds need not be equal, and they can be any number such that the confidence interval not exceed 100%.


## Residual plots

The error term $\varepsilon_i$ has the following assumptions:

* have mean zero; otherwise the forecasts will be systematically biased.
* statistical independence of the errors (in particular, no correlation between consecutive errors in the case of time series data).
* homoscedasticity (constant variance) of the errors.
* normality of the error distribution.  

Plotting the residuals can asses whether (or how much) these assumptions were violated. We will use R to generate residual plots in lesson 2.

![Homoscedasticity](images/Homoscedasticity.png)

![Heteroscedasticity](images/Heteroscedasticity.png)

## Outliers

Observations that take on extreme values compared to the majority can strongky effect the least squares estimators:

$$\hat{\beta}_1=\frac{ \sum_{i=1}^{N}(y_i-\bar{y})(x_i-\bar{x})}{\sum_{i=1}^{N}(x_i-\bar{x})^2} $$

and

$$\hat{\beta}_0=\bar{y}-\hat{\beta}_1\bar{x}, $$

Plotting and occasionally removing outliers and refitting is part of the modeling process.  

## Standard Error of the Regression

How well the model has fitted the data can be thought of as how "tightly" the date fit the regression line. That is, the spread, variance or standard deviation of the residuals.This spread between fitted and actual values is usually known as the *standard error of the regression*:

$$s_e=\sqrt{\frac{1}{N-2}\sum_{i=1}^{N}{e_i^2}}.$$

Here, we divide by $N-2$ because we have estimated two parameters (the intercept and slope) in computing the residuals. Normally, we only need to estimate the mean (i.e., one parameter) when computing a standard deviation. The divisor is always $N$ minus the number of parameters estimated in the calculation.

Note that we can (and should) visualize the predicted vs actual values as this gives more information about the homoscedasticity (constant variance) of the errors.

## The t-statistic and the standard error

The standard error (SE) is the standard deviation of the sampling distribution of a statistic. A sampling distribution is the probability distribution of a given statistic based on a random sample. The dispersion of sample means around the population mean is the standard error. The dispersion of individual observations around the population mean is the standard deviation. The standard error equals the standard deviation divided by the square root of the sample size. As the sample size increases, the dispersion of the sample means clusters more closely around the population mean and the standard error decreases. The standard error is an estimate of the standard deviation of the coefficient. It can be thought of as the spread between fitted and actual values.

$f(x) = \beta_0 + \beta_1 x + \varepsilon . $

Is $H_0: \beta_1 = 0$?

![Standard deviation diagram](images/Standard_deviation_diagram.svg.png)

The t-statistic is the coefficient divided by its standard error. For example, a $\beta_1$ of 38.2 divided by a standard error of 3.4 would give a t value of 11.2.

For a t-statistic high is good. A $\beta_1$ of 38.2 ivided by a standard error of 38.2 would give a t value of 1. A $\beta_1$ of 38.2 ivided by a standard error of 14.1 would give a t value of 2.


$$s_e=\sqrt{\frac{1}{N-2}\sum_{i=1}^{N}{e_i^2}}.$$


If a coefficient is large compared to its standard error, then we can reject the hypothesis that $\beta_1$ = 0. Intuitively we can think of this if the slope is not small and there is a not much spread between fitted and actual values then we can be confident that the true slope $\hat{\beta}_1$ is not 0.

A t-statistic (t value) of greater than 2 in magnitude, corresponds to p-values less than 0.05.

The p-value is a function of the observed sample results (a statistic) that is used for testing a statistical hypothesis. Before the test is performed, a threshold value is chosen, called the significance level of the test, traditionally 5% or 1% and denoted as $\alpha$.  

If the p-value is equal to or smaller than the significance level ($\alpha$), it suggests that the observed data are inconsistent with the assumption that the null hypothesis is true and thus that hypothesis must be rejected (but this does not automatically mean the alternative hypothesis can be accepted as true). When the p-value is calculated correctly, such a test is guaranteed to control the Type I error rate to be no greater than $\alpha$.

from [P-value](https://en.wikipedia.org/wiki/P-value)

## R-squared $R^2$

Regression and R-Squared (2.2) https://youtu.be/Q-TtIPF0fCU

[R-squared](https://en.wikipedia.org/wiki/Coefficient_of_determination)   (coefficient of determination) is a statistical measure of how close the data are to the fitted regression line. the coefficient of determination, denoted $R^2$ or $r^2$ and pronounced "R squared", is a number that indicates the proportion of the variance in the dependent variable that is predictable from the independent variable(s).

R-squared = Explained variation / Total variation

R-squared is always between 0 and 100%:

* 0% (or 0) indicates that the model explains none of the variability of the response data around its mean.  
* 100% (or 1) indicates that the model explains all the variability of the response data around its mean.

The higher the R-squared, the better the model fits your data.

The better the linear regression fits the data in comparison to the simple average (on the left graph), the closer the value of $R^2$ is to 1. The areas of the blue squares represent the squared residuals with respect to the linear regression. The areas of the red squares represent the squared residuals with respect to the average value.]]

A data set has $n$ values marked $y_1$,...,$y_n$ (collectively known as $y_i$ or as a vector $y = [y_1,..., y_n]^T$), each associated with a predicted (or modeled) value $\hat{y}_{1},...,\hat{y}_{n}$.

Define the residuals as as $e_i = y_i − \hat{y}_{i}$ (forming a vector $e$).

If $\bar{y}$ is the mean of the observed data:

$$\bar{y}=\frac{1}{n}\sum_{i=1}^n y_i $$

then the variability of the data set can be measured using three sums of square formulas:

* The total sum of squares (proportional to the variance of the data):

 $$SS_\text{tot}=\sum_i (y_i-\bar{y})^2,$$

* The regression sum of squares, also called the explained sum of squares:

$$SS_\text{reg}=\sum_i (\hat{y}_{i} -\bar{y})^2,$$

* The sum of squares of residuals, also called the residual sum of squares:

$$SS_\text{res}=\sum_i (y_i - \hat{y}_{i})^2=\sum_i e_i^2\,$$

![Correlation_examples](images/r-squared_A.png)

The most general definition of the coefficient of determination is

$$R^2 \equiv 1 - {SS_{\rm residuals}\over SS_{\rm total}}.\,$$

$R^2$ can be thought of as *the proportion of variation in the forecast variable that is accounted for (or explained) by the regression model**

In the definition of $R^2$, $0 \geq R^2 \geq 1$ as is similar to he value of $r^2$ (the square of the pearson correlation between $f(x)$ and $x$.


![Correlation_examples](images/r-squared.png)

Image credit: [http://www.rapidinsightinc.com/brushing-r-squared/](http://www.rapidinsightinc.com/brushing-r-squared/)


```python
def random_distributions(n=555):
  mu, sigma, scale = 5.5, 2*np.sqrt(2), 0.5
  a = np.random.normal(mu, sigma, n)
  b = np.random.normal(mu, sigma, n)
  c = np.linspace(np.amin(a),np.amax(a),n)
  d = mu + (sigma*c) + np.random.normal(mu*scale, sigma*scale, n)
  df = pd.DataFrame({'A' :a, 'B' :b,'C' :c,'D' :d})
  return df
```


```python
n=555
df=random_distributions(n)
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9.423756</td>
      <td>7.709713</td>
      <td>-2.867302</td>
      <td>-0.516642</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.450986</td>
      <td>5.104608</td>
      <td>-2.835068</td>
      <td>-0.239419</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.961451</td>
      <td>4.122580</td>
      <td>-2.802834</td>
      <td>3.515491</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.633739</td>
      <td>7.753216</td>
      <td>-2.770600</td>
      <td>1.186225</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.602576</td>
      <td>4.688394</td>
      <td>-2.738367</td>
      <td>-0.059730</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>555.000000</td>
      <td>555.000000</td>
      <td>555.000000</td>
      <td>555.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.426252</td>
      <td>5.308850</td>
      <td>6.061458</td>
      <td>25.502693</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.878144</td>
      <td>2.911776</td>
      <td>5.168977</td>
      <td>14.662229</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.867302</td>
      <td>-3.523335</td>
      <td>-2.867302</td>
      <td>-0.602034</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.435052</td>
      <td>3.286059</td>
      <td>1.597078</td>
      <td>13.013597</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.296571</td>
      <td>5.377445</td>
      <td>6.061458</td>
      <td>25.662476</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.271397</td>
      <td>7.285304</td>
      <td>10.525837</td>
      <td>37.843537</td>
    </tr>
    <tr>
      <th>max</th>
      <td>14.990217</td>
      <td>14.047461</td>
      <td>14.990217</td>
      <td>52.884133</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = df["A"]
y = df["B"]
fit= np.polyfit(X, y, 1)
fit_fn = np.poly1d(fit)
plt.plot(X,y, 'yo', X, fit_fn(X), '--k')
x2 = np.linspace(np.amin(X),np.amax(X),n)
y2 = np.ones(n)*np.mean(y)
plt.plot(x2, y2, lw=3, color="purple")
plt.show()
```


    
![png](output_20_0.png)
    


SSR is the "regression sum of squares" and quantifies how far the estimated sloped regression line, $\hat{y}_{i}$, is from the horizontal "no relationship line," the sample mean or $\bar{y}$ (the straight green line).

$$SS_\text{regression}=\sum_i (\hat{y}_{i} -\bar{y})^2,$$

Note:  $\hat{y}_{i}$ is the predicted value on the regression line.

In a poor fit $SS_\text{regression}$ is small in relation to the variance around $\bar{y}$ (i.e. the total variance).


```python
X = df["C"]
y = df["D"]
fit= np.polyfit(X, y, 1)
fit_fn = np.poly1d(fit)
plt.plot(X,y, 'yo', X, fit_fn(X), '--k')
x2 = np.linspace(np.amin(X),np.amax(X),n)
y2 = np.ones(n)*np.mean(y)
plt.plot(x2, y2, lw=3, color="purple")
plt.show()
```


    
![png](output_22_0.png)
    


SSE is the "error sum of squares" or "least square error" and quantifies how much the data points, $y_i$, vary around the estimated regression line, $\hat{y}_{i}$.

The sum of squares of residuals, also called the residual sum of squares, error sum of squares and least square error:

$$SS_\text{residuals}=\sum_i (y_i - \hat{y}_{i})^2=\sum_i e_i^2\,$$

NOTE: That if the $SS_\text{residuals}$ is small, it is good. One wants the residual error to be small.

SSTO ($SS_\text{total}$) is the "total sum of squares" and quantifies how much the data points, $y_i$, vary around their mean, $\bar{y}$.

$$SS_\text{total}=\sum_i (y_i - \bar{y}_{i})^2=\sum_i e_i^2\,$$

Note that $SS_\text{total} = SS_\text{regression} + SS_\text{residuals}$.

The most general definition of the coefficient of determination is

$$R^2 \equiv {SS_{\rm regression}\over SS_{\rm total}}  \equiv 1 - {SS_{\rm residuals}\over SS_{\rm total}} $$

Therefore when $SS_{\rm residuals}$ is small in relation to $SS_{\rm total}$ then $R^2$ will be high (near 1).

## F-test

In regression it is known that

$$ Var(residuals) \over Var(errors) $$

$f(x) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 ... + \beta_n x_n + \varepsilon . $

Is $H_0: \beta_1 = \beta_2 = \beta_3  ... = \beta_n = 0$?

$$
H_{0}: β_{1}= β_{2}=….= β_{n} = 0
$$
$$
H_{1}: \quad at \quad least \quad one \quad pair \quad  β_{j}≠ β_{j'}
$$
The o

follows a [Chi-squared](https://en.wikipedia.org/wiki/Chi-squared_distribution) $\chi^2$ distribution.

The p-values and t-stats assess the fit of individual parameters. It would be nice to have a measure of fit that assesses the overall regression. The F-stat is a measure for the regression as a whol

An [F-test](https://en.wikipedia.org/wiki/F-test) is any statistical test in which the test statistic has an F-distribution under the null hypothesis. Like a t-statistic, or a p-value it provides an estimate of whether one should accept or reject the null hypothesis. The F-test is sensitive to non-normality (as is a t-statistic) but is appropriate under the assumptions of normality and [homoscedasticity](https://en.wikipedia.org/wiki/Homoscedasticity).

In a sum of squares due to lack of fit is one of the components of a partition of the sum of squares in an analysis of variance, used in the numerator in an F-test of the null hypothesis that says that a proposed whole model fits well.

In order for the lack-of-fit sum of squares to differ from the Residual sum of squares, there must be more than one value of the response variable for at least one of the values of the set of predictor variables.  For example, consider fitting a line

$$ y = \alpha x + \beta \, $$

by the method of least squares.  One takes as estimates of ''α'' and ''β'' the values that minimize the sum of squares of residuals, i.e., the sum of squares of the differences between the observed ''y''-value and the fitted ''y''-value.  To have a lack-of-fit sum of squares that differs from the residual sum of squares, one must observe more than one ''y''-value for each of one or more of the ''x''-values.  One then partitions the "sum of squares due to error", i.e., the sum of squares of residuals, into two components:

$$
\begin{align}
F & = \frac{ \text{lack-of-fit sum of squares} /\text{degrees of freedom} }{\text{pure-error sum of squares} / \text{degrees of freedom} } \\[8pt]
& = \frac{\left.\sum_{i=1}^n n_i \left( \overline Y_{i\bullet} - \widehat Y_i \right)^2\right/ (n-p)}{\left.\sum_{i=1}^n \sum_{j=1}^{n_i} \left(Y_{ij} - \overline Y_{i\bullet}\right)^2 \right/ (N - n)}
\end{align}
$$

has an *F-distribution* with the corresponding number of degrees of freedom in the numerator and the denominator, provided that the model is correct. If the model is wrong, then the probability distribution of the denominator is still as stated above, and the numerator and denominator are still independent.  But the numerator then has a *noncentral chi-squared distribution*, and consequently the quotient as a whole has a *non-central F-distribution*. The F-distribution, also known as Snedecor's F distribution or the Fisher–Snedecor distribution (after Ronald Fisher and George W. Snedecor) is a continuous probability distribution that arises frequently as the null distribution of a test statistic, most notably in the analysis of variance.

The assumptions of normal distribution of errors and independence can be shown to entail that this lack-of-fit test is the *likelihood-ratio test* of this null hypothesis.
  
A high F-stat is good.

## Leverage

A data point has high leverage if it has "extreme" predictor x values. With a single predictor, an extreme x value is simply one that is particularly high or low.  

Leverage and Influential Points in Simple Linear Regression https://youtu.be/xc_X9GFVuVU


## Influence

A data point is influential if it unduly influences any part of a regression analysis. Influence is high leverage and high residuals and is often meaured by statistics like the Cook's distance.

![Leverage vs Influenc](images/leverage_vs_influence.png)

Image credit:  [Leverage and Influential Points in Simple Linear Regression https://youtu.be/xc_X9GFVuVU](Leverage and Influential Points in Simple Linear Regression https://youtu.be/xc_X9GFVuVU)




## K-fold cross validation

In general  to assess the fit of the model one can use summary measures of goodness of fit (such as $R^2$) or by assessing the predictive ability of the model (using  k-fold cross-validation). We'd also like to deterime if there’s any observations that do not fit the model or that have an undue influence on the model.

K-fold cross validation is a simple, intuitive way to estimate prediction error.  K-fold cross-validation, which partitions the data into $k$ equally sized segments (called ‘folds’). One fold is held out for validation while the other $k-1$ folds are used to train the model and then used to predict the target variable in our testing data. This process is repeated $k$ times, with the performance of each model in predicting the hold-out set being tracked using a performance metric such as accuracy. In the case of regression that would be the residuals of the test set based on the linear model generated for the training set.

## Residual Plots

The error term $\varepsilon_i$ has the following assumptions:

* have mean zero; otherwise the forecasts will be systematically biased.
* statistical independence of the errors (in particular, no correlation between consecutive errors in the case of time series data).
* homoscedasticity (constant variance) of the errors.
* normality of the error distribution.  

The typical plots are:

* _Residuals vs Fitted_  

the residuals and the fitted values should be uncorrelated in a [homoscedastic](https://en.wikipedia.org/wiki/Homoscedasticity) linear model with normally distributed errors. There should not be a dependency between the residuals and the fitted values,  

![Residuals vs Fitted](images/Residuals_vs_Fitted.png)


* _Residuls vs Normal_  

This is a [Q–Q plot](https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot) to check if the residuls are normal (i.e. normality of the error distribution.)    

![Residuals vs Fitted](images/Residuals_vs_Normal.png)

* _Standardized Residuals vs Fitted Values_  

[standardized residuals](https://en.wikipedia.org/wiki/Studentized_residual) means every residual plot you look at with any model is on the same standardized y-axis. A standardized (or studentized) residual is the quotient resulting from the division of a residual by an estimate of its standard deviation. This makes it easier to compare many residul plots. This process is also called *studentizing* (after [William Sealey Gosset](https://en.wikipedia.org/wiki/William_Sealy_Gosset), who wrote under the pseudonym Student).

The key reason for studentizing is that, in regression analysis of a multivariate distribution, the variances of the residuals at different input variable values may differ, even if the variances of the errors at these different input variable values are equal.

![Standardized Residuals vs Fitted Values](images/Standardized_Residuals_vs_Fitted_Values.png)

* _Residuals vs Leverage_  

We use leverage to check for outliers. To understand [leverage](https://en.wikipedia.org/wiki/Leverage_%28statistics%29), recognize that simple linear regression fits a line that will pass through the center of your data. High-leverage points are those observations, if any, made at extreme or outlying values of the independent variables such that the lack of neighboring observations means that the fitted regression model will pass close to that particular observation.

To think of the leverage of a point consider how the slope might change if the model were fit without the data point in question. A common way to estimate of the influence of a data point is [Cook's distance or Cook's D](https://en.wikipedia.org/wiki/Cook%27s_distance)

$$ D_i = \frac{\sum_{j = 1}^n (\hat{y}_{j(i)} - \hat{y}_j)^2}{2S^2} $$

where $\hat{y}_{j(i)}$ is the $j^{th}$ fitted value based on the fit with the $i^{th}$ point removed. ${S^2} $is the [Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)

An alternate form of Cook's distance:

$$ D_i = \frac{r_i^2}{2} \frac{h_{ii}}{1 - h_{ii}} $$

To be influential a point must:

Have high leverage $h_{ii}$ and
Have a high standardized residual $r_i$

Analyists often look for and remove high leverage points and re-fit a model.

![Residuals vs Leverage](images/Residuals_vs_Leverage.png)

## Multi-colinearity

In statistics, multicollinearity (also collinearity) is a phenomenon in which two or more predictor variables in a multiple regression model are highly correlated, meaning that one can be linearly predicted from the others with a non-trivial degree of accuracy. In this situation the coefficient estimates of the multiple regression may change erratically in response to small changes in the model or the data. Multicollinearity does not reduce the predictive power or reliability of the model as a whole, at least within the sample data set; it only affects calculations regarding individual predictors. That is, a multiple regression model with correlated predictors can indicate how well the entire bundle of predictors predicts the outcome variable, but it may not give valid results about any individual predictor, or about which predictors are redundant with respect to others.

- from [multicollinearity - Wikipedia](https://en.wikipedia.org/wiki/Multicollinearity)


## Extensions of the Linear Model  

Removing the additive assumption: interactions and
nonlinearity
Interactions:
- In our previous analysis of the Advertising data, we assumed that the effect on sales of increasing one advertising medium is independent of the amount spent on the other media.
- For example, the linear model
\begin{equation}\text { sales }=\beta_{0}+\beta_{1} \times \mathrm{TV}+\beta_{2} \times \mathrm{radio}+\beta_{3} \times \text { newspaper }\end{equation}
states that the average effect on sales of a one-unit
increase in TV is always β1, regardless of the amount spent
on radio.
- But suppose that spending money on radio advertising actually increases the effectiveness of TV advertising, so that the slope term for TV should increase as radio increases.
- In this situation, given a fixed budget of 100, 000, spending half on radio and half on TV may increase sales more than allocating the entire amount to either TV or to radio.
- In marketing, this is known as a synergy effect, and in statistics it is referred to as an interaction effect.
- When levels of either TV or radio are low, then the true sales are lower than predicted by the linear model. But when advertising is split between the two media, then the model tends to underestimate sales.

## Interaction Variables  

In statistics, an interaction may arise when considering the relationship among three or more variables, and describes a situation in which the effect of one causal variable on an outcome depends on the state of a second causal variable (that is, when effects of the two causes are not additive)

We create these in linear regression through product terms. That is we multiply two independent variables together and use that product as another independent variable.


### Modelling interactions — Advertising data   

Model takes the form
\begin{equation}\begin{aligned}
\text { sales } &=\beta_{0}+\beta_{1} \times \mathrm{TV}+\beta_{2} \times \text { radio }+\beta_{3} \times(\text { radio } \times \mathrm{TV})+\epsilon \\
&=\beta_{0}+\left(\beta_{1}+\beta_{3} \times \mathrm{radio}\right) \times \mathrm{TV}+\beta_{2} \times \text { radio }+\epsilon
\end{aligned}\end{equation}  


```python
results2 = smf.ols('Sales ~ TV + Radio + TV*Radio', data=df_adv).fit()
results2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>Sales</td>      <th>  R-squared:         </th> <td>   0.968</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.967</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1963.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 14 Sep 2020</td> <th>  Prob (F-statistic):</th> <td>6.68e-146</td>
</tr>
<tr>
  <th>Time:</th>                 <td>02:10:10</td>     <th>  Log-Likelihood:    </th> <td> -270.14</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   200</td>      <th>  AIC:               </th> <td>   548.3</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   196</td>      <th>  BIC:               </th> <td>   561.5</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    6.7502</td> <td>    0.248</td> <td>   27.233</td> <td> 0.000</td> <td>    6.261</td> <td>    7.239</td>
</tr>
<tr>
  <th>TV</th>        <td>    0.0191</td> <td>    0.002</td> <td>   12.699</td> <td> 0.000</td> <td>    0.016</td> <td>    0.022</td>
</tr>
<tr>
  <th>Radio</th>     <td>    0.0289</td> <td>    0.009</td> <td>    3.241</td> <td> 0.001</td> <td>    0.011</td> <td>    0.046</td>
</tr>
<tr>
  <th>TV:Radio</th>  <td>    0.0011</td> <td> 5.24e-05</td> <td>   20.727</td> <td> 0.000</td> <td>    0.001</td> <td>    0.001</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>128.132</td> <th>  Durbin-Watson:     </th> <td>   2.224</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>1183.719</td> 
</tr>
<tr>
  <th>Skew:</th>          <td>-2.323</td>  <th>  Prob(JB):          </th> <td>9.09e-258</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>13.975</td>  <th>  Cond. No.          </th> <td>1.80e+04</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.8e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.





#### Interpretation
- The results in this table suggests that interactions are important.
- The p-value for the interaction term TV×radio is extremely low, indicating that there is strong evidence for HA : β3 6= 0.
- The R2 for the interaction model is 96.8%, compared to only 89.7% for the model that predicts sales using TV and radio without an interaction term.
- This means that (96.8 − 89.7)/(100 − 89.7) = 69% of the variability in sales that remains after fitting the additive model has been explained by the interaction term.
- The coefficient estimates in the table suggest that an increase in TV advertising of 1, 000 is associated with increased sales of $(\hat{\beta}_{1} + \hat{\beta}_{3} × radio) × 1000 = 19 + 1.1 × radio$ units.
- An increase in radio advertising of 1, 000 will be associated with an increase in sales of
$(\hat{\beta}_{2} + \hat{\beta}_{3} × TV) × 1000 = 29 + 1.1 × TV$ units.


```python

```

## Credit Card Data (Qualitative Predictors)

Let's import a data set with categorical independent variables so that we understand how to use them and the difference between dummy variables and one-hot encoding,




```python
credit = pd.read_csv('https://raw.githubusercontent.com/nikbearbrown/Google_Colab/master/data/credit.csv')
print(credit.shape)
credit.head()
```

    (400, 11)
    




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
      <th>Income</th>
      <th>Limit</th>
      <th>Rating</th>
      <th>Cards</th>
      <th>Age</th>
      <th>Education</th>
      <th>Gender</th>
      <th>Student</th>
      <th>Married</th>
      <th>Ethnicity</th>
      <th>Balance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.891</td>
      <td>3606</td>
      <td>283</td>
      <td>2</td>
      <td>34</td>
      <td>11</td>
      <td>Male</td>
      <td>No</td>
      <td>Yes</td>
      <td>Caucasian</td>
      <td>333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>106.025</td>
      <td>6645</td>
      <td>483</td>
      <td>3</td>
      <td>82</td>
      <td>15</td>
      <td>Female</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Asian</td>
      <td>903</td>
    </tr>
    <tr>
      <th>2</th>
      <td>104.593</td>
      <td>7075</td>
      <td>514</td>
      <td>4</td>
      <td>71</td>
      <td>11</td>
      <td>Male</td>
      <td>No</td>
      <td>No</td>
      <td>Asian</td>
      <td>580</td>
    </tr>
    <tr>
      <th>3</th>
      <td>148.924</td>
      <td>9504</td>
      <td>681</td>
      <td>3</td>
      <td>36</td>
      <td>11</td>
      <td>Female</td>
      <td>No</td>
      <td>No</td>
      <td>Asian</td>
      <td>964</td>
    </tr>
    <tr>
      <th>4</th>
      <td>55.882</td>
      <td>4897</td>
      <td>357</td>
      <td>2</td>
      <td>68</td>
      <td>16</td>
      <td>Male</td>
      <td>No</td>
      <td>Yes</td>
      <td>Caucasian</td>
      <td>331</td>
    </tr>
  </tbody>
</table>
</div>




```python
credit.describe()
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
      <th>Income</th>
      <th>Limit</th>
      <th>Rating</th>
      <th>Cards</th>
      <th>Age</th>
      <th>Education</th>
      <th>Balance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>45.218885</td>
      <td>4735.600000</td>
      <td>354.940000</td>
      <td>2.957500</td>
      <td>55.667500</td>
      <td>13.450000</td>
      <td>520.015000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>35.244273</td>
      <td>2308.198848</td>
      <td>154.724143</td>
      <td>1.371275</td>
      <td>17.249807</td>
      <td>3.125207</td>
      <td>459.758877</td>
    </tr>
    <tr>
      <th>min</th>
      <td>10.354000</td>
      <td>855.000000</td>
      <td>93.000000</td>
      <td>1.000000</td>
      <td>23.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>21.007250</td>
      <td>3088.000000</td>
      <td>247.250000</td>
      <td>2.000000</td>
      <td>41.750000</td>
      <td>11.000000</td>
      <td>68.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>33.115500</td>
      <td>4622.500000</td>
      <td>344.000000</td>
      <td>3.000000</td>
      <td>56.000000</td>
      <td>14.000000</td>
      <td>459.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>57.470750</td>
      <td>5872.750000</td>
      <td>437.250000</td>
      <td>4.000000</td>
      <td>70.000000</td>
      <td>16.000000</td>
      <td>863.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>186.634000</td>
      <td>13913.000000</td>
      <td>982.000000</td>
      <td>9.000000</td>
      <td>98.000000</td>
      <td>20.000000</td>
      <td>1999.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(credit)
```




    <seaborn.axisgrid.PairGrid at 0x7f76b381a438>




    
![png](output_36_1.png)
    


## Qualitative Predictors  

Example: investigate differences in credit card balance between
males and females, ignoring the other variables. We create a
new variable  

\begin{equation}x_{i}=\left\{\begin{array}{ll}
1 & \text { if } i \text { th person is female } \\
0 & \text { if } i \text { th person is male. }
\end{array}\right.\end{equation}

Resulting model
\begin{equation}y_{i}=\beta_{0}+\beta_{1} x_{i}+\epsilon_{i}=\left\{\begin{array}{ll}
\beta_{0}+\beta_{1}+\epsilon_{i} & \text { if } i \text { th person is female } \\
\beta_{0}+\epsilon_{i} & \text { if } i \text { th person is male. }
\end{array}\right.\end{equation}

Intrepretation?


```python
# For categorical variable consideration
credit['Female'] = (credit.Gender == 'Female').astype(int)
credit.head()
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
      <th>Income</th>
      <th>Limit</th>
      <th>Rating</th>
      <th>Cards</th>
      <th>Age</th>
      <th>Education</th>
      <th>Gender</th>
      <th>Student</th>
      <th>Married</th>
      <th>Ethnicity</th>
      <th>Balance</th>
      <th>Female</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.891</td>
      <td>3606</td>
      <td>283</td>
      <td>2</td>
      <td>34</td>
      <td>11</td>
      <td>Male</td>
      <td>No</td>
      <td>Yes</td>
      <td>Caucasian</td>
      <td>333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>106.025</td>
      <td>6645</td>
      <td>483</td>
      <td>3</td>
      <td>82</td>
      <td>15</td>
      <td>Female</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Asian</td>
      <td>903</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>104.593</td>
      <td>7075</td>
      <td>514</td>
      <td>4</td>
      <td>71</td>
      <td>11</td>
      <td>Male</td>
      <td>No</td>
      <td>No</td>
      <td>Asian</td>
      <td>580</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>148.924</td>
      <td>9504</td>
      <td>681</td>
      <td>3</td>
      <td>36</td>
      <td>11</td>
      <td>Female</td>
      <td>No</td>
      <td>No</td>
      <td>Asian</td>
      <td>964</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>55.882</td>
      <td>4897</td>
      <td>357</td>
      <td>2</td>
      <td>68</td>
      <td>16</td>
      <td>Male</td>
      <td>No</td>
      <td>Yes</td>
      <td>Caucasian</td>
      <td>331</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
results_g = smf.ols('Balance ~ Female', data=credit).fit()
results_g.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Balance</td>     <th>  R-squared:         </th> <td>   0.000</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>  -0.002</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>  0.1836</td>
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 14 Sep 2020</td> <th>  Prob (F-statistic):</th>  <td> 0.669</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>02:21:58</td>     <th>  Log-Likelihood:    </th> <td> -3019.3</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   400</td>      <th>  AIC:               </th> <td>   6043.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   398</td>      <th>  BIC:               </th> <td>   6051.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>  509.8031</td> <td>   33.128</td> <td>   15.389</td> <td> 0.000</td> <td>  444.675</td> <td>  574.931</td>
</tr>
<tr>
  <th>Female</th>    <td>   19.7331</td> <td>   46.051</td> <td>    0.429</td> <td> 0.669</td> <td>  -70.801</td> <td>  110.267</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>28.438</td> <th>  Durbin-Watson:     </th> <td>   1.940</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  27.346</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.583</td> <th>  Prob(JB):          </th> <td>1.15e-06</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.471</td> <th>  Cond. No.          </th> <td>    2.66</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



#### Qualitative predictors with multiple levels  

- With more than two levels, we create additional dummy variables. For example, for the ethnicity variable we create two dummy variables. The first could be
\begin{equation}x_{i1}=\left\{\begin{array}{ll}
1 & \text { if } i \text { th person is Asian } \\
0 & \text { if } i \text { th person is not Asian }
\end{array}\right.\end{equation}

and the second could be

\begin{equation}x_{i2}=\left\{\begin{array}{ll}
1 & \text { if } i \text { th person is Caucasian } \\
0 & \text { if } i \text { th person is not Caucasian }
\end{array}\right.\end{equation}

- Then both of these variables can be used in the regression equation, in order to obtain the model
\begin{equation}y_{i}=\beta_{0}+\beta_{1} x_{i}+\beta_{2} x_{i2}+\epsilon_{i}=\left\{\begin{array}{ll}
\beta_{0}+\beta_{1}+\epsilon_{i} & \text { if } i \text { th person is Asian } \\
\beta_{0}+\beta_{2}+\epsilon_{i} & \text { if } i \text { th person is Causasian } \\
\beta_{0}+\epsilon_{i} & \text { if } i \text { th person is AA }
\end{array}\right.\end{equation}
- There will always be one fewer dummy variable than the number of levels. The level with no dummy variable — African American in this example — is known as the $\textit{baseline}$.



```python
credit_2 = pd.get_dummies(credit,columns=['Ethnicity'])
credit_2.head()
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
      <th>Income</th>
      <th>Limit</th>
      <th>Rating</th>
      <th>Cards</th>
      <th>Age</th>
      <th>Education</th>
      <th>Gender</th>
      <th>Student</th>
      <th>Married</th>
      <th>Balance</th>
      <th>Ethnicity_African American</th>
      <th>Ethnicity_Asian</th>
      <th>Ethnicity_Caucasian</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.891</td>
      <td>3606</td>
      <td>283</td>
      <td>2</td>
      <td>34</td>
      <td>11</td>
      <td>Male</td>
      <td>No</td>
      <td>Yes</td>
      <td>333</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>106.025</td>
      <td>6645</td>
      <td>483</td>
      <td>3</td>
      <td>82</td>
      <td>15</td>
      <td>Female</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>903</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>104.593</td>
      <td>7075</td>
      <td>514</td>
      <td>4</td>
      <td>71</td>
      <td>11</td>
      <td>Male</td>
      <td>No</td>
      <td>No</td>
      <td>580</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>148.924</td>
      <td>9504</td>
      <td>681</td>
      <td>3</td>
      <td>36</td>
      <td>11</td>
      <td>Female</td>
      <td>No</td>
      <td>No</td>
      <td>964</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>55.882</td>
      <td>4897</td>
      <td>357</td>
      <td>2</td>
      <td>68</td>
      <td>16</td>
      <td>Male</td>
      <td>No</td>
      <td>Yes</td>
      <td>331</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
dummies = pd.get_dummies(credit['Ethnicity']).rename(columns=lambda x: 'Ethnicity_' + str(x))
dummies.head()
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
      <th>Ethnicity_African American</th>
      <th>Ethnicity_Asian</th>
      <th>Ethnicity_Caucasian</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
credit= pd.concat([credit, dummies], axis=1)
credit.head()
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
      <th>Income</th>
      <th>Limit</th>
      <th>Rating</th>
      <th>Cards</th>
      <th>Age</th>
      <th>Education</th>
      <th>Gender</th>
      <th>Student</th>
      <th>Married</th>
      <th>Ethnicity</th>
      <th>Balance</th>
      <th>Ethnicity_African American</th>
      <th>Ethnicity_Asian</th>
      <th>Ethnicity_Caucasian</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.891</td>
      <td>3606</td>
      <td>283</td>
      <td>2</td>
      <td>34</td>
      <td>11</td>
      <td>Male</td>
      <td>No</td>
      <td>Yes</td>
      <td>Caucasian</td>
      <td>333</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>106.025</td>
      <td>6645</td>
      <td>483</td>
      <td>3</td>
      <td>82</td>
      <td>15</td>
      <td>Female</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Asian</td>
      <td>903</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>104.593</td>
      <td>7075</td>
      <td>514</td>
      <td>4</td>
      <td>71</td>
      <td>11</td>
      <td>Male</td>
      <td>No</td>
      <td>No</td>
      <td>Asian</td>
      <td>580</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>148.924</td>
      <td>9504</td>
      <td>681</td>
      <td>3</td>
      <td>36</td>
      <td>11</td>
      <td>Female</td>
      <td>No</td>
      <td>No</td>
      <td>Asian</td>
      <td>964</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>55.882</td>
      <td>4897</td>
      <td>357</td>
      <td>2</td>
      <td>68</td>
      <td>16</td>
      <td>Male</td>
      <td>No</td>
      <td>Yes</td>
      <td>Caucasian</td>
      <td>331</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
results_e = smf.ols('Balance ~ Ethnicity', data=credit).fit()
results_e.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Balance</td>     <th>  R-squared:         </th> <td>   0.000</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>  -0.005</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td> 0.04344</td>
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 14 Sep 2020</td> <th>  Prob (F-statistic):</th>  <td> 0.957</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>02:44:21</td>     <th>  Log-Likelihood:    </th> <td> -3019.3</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   400</td>      <th>  AIC:               </th> <td>   6045.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   397</td>      <th>  BIC:               </th> <td>   6057.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
             <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>              <td>  531.0000</td> <td>   46.319</td> <td>   11.464</td> <td> 0.000</td> <td>  439.939</td> <td>  622.061</td>
</tr>
<tr>
  <th>Ethnicity[T.Asian]</th>     <td>  -18.6863</td> <td>   65.021</td> <td>   -0.287</td> <td> 0.774</td> <td> -146.515</td> <td>  109.142</td>
</tr>
<tr>
  <th>Ethnicity[T.Caucasian]</th> <td>  -12.5025</td> <td>   56.681</td> <td>   -0.221</td> <td> 0.826</td> <td> -123.935</td> <td>   98.930</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>28.829</td> <th>  Durbin-Watson:     </th> <td>   1.946</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  27.395</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.581</td> <th>  Prob(JB):          </th> <td>1.13e-06</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.460</td> <th>  Cond. No.          </th> <td>    4.39</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



## Exercise  

*Question 1*  

Given the linear model below, what is/are the null hypotheses?

$$ Sales = \beta_0 + \beta_1 TV + \beta_2 Radio + \beta_3 Newspaper + \varepsilon . $$



*Answer 1*

There are 3 different null hypotheses for each of TV, Radio and Newspaper each testing whether there is a relationship from that variable to Sales given that the other two variables are held constant. From this model we can reject the null hypotheses that both TV and Radio have no correspondence with sales. We fail to reject the null hypotheses that Newspaper advertising is related to Sales.


*Question 2*  

Answer the following questions:

a) For a linear model how will increasing the model complexity affect the fit on the training data?

b) For a linear model how will increasing the model complexity affect the fit on the testing data?


*Answer 2*  

a) For training data, the RSS always decreases as model complexity increases so the cubic model will have lower RSS.

b) For testing data, too much complexity may overfit the data causing the testing error to increase.



## Quiz (5 points)  Make a contribution to this notebook

For 5 points. Make a contribution to this notebook. This will be explained in class.  
