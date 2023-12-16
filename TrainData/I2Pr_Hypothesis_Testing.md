# Hypothesis Testing

### Hypothesis Testing

A **statistical hypothesis**, sometimes called **confirmatory data
analysis**, is a hypothesis that is testable on the basis of
observing a process that is modeled via a set of random
variables. A **statistical hypothesis test** is a method of
statistical inference. Commonly, two statistical data sets are compared, or a data set obtained by sampling is compared against a synthetic data set from an idealized model. A hypothesis is proposed for the statistical relationship between the two data sets, and this is compared as an alternative to an idealized null hypothesis that proposes no relationship between two data sets. The comparison is deemed statistically significant if the relationship between the data sets would be an unlikely realization of the null hypothesis according to a threshold probability---the significance level. Hypothesis tests are used in determining what outcomes of a study would lead to a rejection of the null hypothesis for a pre-specified level of significance. The process of distinguishing between the null hypothesis and the alternative hypothesis is aided by identifying two conceptual types of errors (type 1 & type 2), and by specifying parametric limits on e.g. how much type 1 error will be permitted.

An alternative framework for statistical hypothesis testing is to specify a set of statistical models, one for each candidate hypothesis, and then use model selection techniques to choose the most appropriate model. The most common selection techniques are based on either Akaike information criterion or Bayes factor.

Confirmatory data analysis can be contrasted with exploratory data
analysis, which may not have pre-specified hypotheses.

![An Introduction to Hypothesis Testing](http://nikbearbrown.com/YouTube/MachineLearning/IMG/An_Introduction_to_Hypothesis_Testing.png)

An Introduction to Hypothesis Testing [https://youtu.be/tTeMYuS87oU](https://youtu.be/tTeMYuS87oU)     


![Z Tests for One Mean: Introduction](http://nikbearbrown.com/YouTube/MachineLearning/IMG/Z_Tests_for_One_Mean_Introduction.png)

Z Tests for One Mean: Introduction [https://youtu.be/pGv13jvnjKc](https://youtu.be/pGv13jvnjKc)    

![Z Tests for One Mean: The Rejection Region Approach](http://nikbearbrown.com/YouTube/MachineLearning/IMG/Z_Tests_for_One_Mean_The_Rejection_Region_Approach.png)

Z Tests for One Mean: The Rejection Region Approach [https://youtu.be/60x86lYtWI4](https://youtu.be/60x86lYtWI4)   

![Z Tests for One Mean: The p-value](http://nikbearbrown.com/YouTube/MachineLearning/IMG/Z_Tests_for_One_Mean_The_p-value.png)


Z Tests for One Mean: The p-value [https://youtu.be/m6sGjWz2CPg](https://youtu.be/m6sGjWz2CPg)  


```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.testing as tm
from scipy import stats
import seaborn as sns
import time
np.random.seed(seed=int(time.time()))

# Make plots larger
plt.rcParams['figure.figsize'] = (15, 9)
```

## Hypothesis testing

A hypothesis is proposed for the statistical relationship between the two data sets, and this is compared as an alternative to an idealized null hypothesis that proposes no relationship between two data sets. The comparison is deemed statistically significant if the relationship between the data sets would be an unlikely realization of the null hypothesis according to a threshold probability—the significance level. Hypothesis tests are used in determining what outcomes of a study would lead to a rejection of the null hypothesis for a pre-specified level of significance.

[Hypothesis tests](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing) assume the thing you want to disprove, and then to look for evidence that the assumption is wrong. In this case, we assume that there is no difference between $\bar{x_1}$ and $\bar{x_2}$ (i.e. the mean of one distribution versus another) This is called the *null hypothesis* and is stated as

$$H_0: \bar{x_1} = \bar{x_2}$$

If $\bar{x_1}$ is very different from $\bar{x_2}$ we conclude that the null hypothesis is incorrect and that the evidence suggests there really is a difference between $\bar{x_1}$ and $\bar{x_2}$.

There are many hypothesis tests that can be used to test whether there is a difference between $\bar{x_1}$ and $\bar{x_2}$:

* Student’s T-Tests
* One-Sample T-Test
* Two-Sample T-Test
* Paired T-Test
* Wilcoxon Rank-Sum Test
* Analysis of Variance (ANOVA)
* Kruskal-Wallis Test

We will discuss these more in the module on hypothesis testing.

## P-value

To determine how big the difference between  $\bar{x_1}$ and  $\bar{x_2}$ must be before we would reject the null hypothesis, we calculate the probability of obtaining a value of $\bar{x_2}$ as large as we have calculated if the null hypothesis were true. This probability is known as the *P-value*.

In statistics, the p-value is a function of the observed sample results (a statistic) that is used for testing a statistical hypothesis. Before the test is performed, a threshold value is chosen, called the significance level of the test, traditionally 5% or 1% and denoted as $\alpha$.  

If the p-value is equal to or smaller than the significance level ($\alpha$), it suggests that the observed data are inconsistent with the assumption that the null hypothesis is true and thus that hypothesis must be rejected (but this does not automatically mean the alternative hypothesis can be accepted as true). When the p-value is calculated correctly, such a test is guaranteed to control the Type I error rate to be no greater than $\alpha$.

from [P-value](https://en.wikipedia.org/wiki/P-value)

A is **p-value** is the probability of observing a test statistic equally or more extreme than the one you observed, assuming the hypothesis you are testing is true.
    

## Confidence intervals

In statistics, a confidence interval (CI) is a type of interval estimate of a population parameter. It provides an interval estimate for lower or upper confidence bounds. For $\beta_1$, usually referred to as a *confidence interval* and is typically +/-0.5% (a 99% confidence interval),+/-1% (a 98% confidence interval),+/-2.5% (a 95% confidence interval) or +/-5% (a 90% confidence interval). The lower and upper confidence bounds need not be equal, and they can be any number such that the confidence interval not exceed 100%.


## The t-distribution

In probability and statistics, Student's t-distribution (or simply the t-distribution) is any member of a family of continuous probability distributions that arises when estimating the mean of a normally distributed population in situations where the sample size is small and population standard deviation is unknown. Whereas a normal distribution describes a full population, t-distributions describe samples drawn from a full population; accordingly, the t-distribution for each sample size is different, and the larger the sample, the more the distribution resembles a normal distribution.
The t-distribution plays a role in a number of widely used statistical analyses, including the Student's t-test for assessing the statistical significance of the difference between two sample means, the construction of confidence intervals for the difference between two population means, and in linear regression analysis. The Student's t-distribution also arises in the Bayesian analysis of data from a normal family.

- from [The t-distribution - Wikipedia)](https://en.wikipedia.org/wiki/Student%27s_t-distribution)

When the CLT does not apply (i.e. as the number of samples is large), there is another option that does not rely on large samples When a the original population from which a random variable, say $Y$, is sampled is normally distributed with mean 0 then we can calculate the distribution of


number of variants. In its common form, the random variables must be identically distributed.



$$
\sqrt{N} \frac{\bar{Y}}{s_Y}
$$


![image Student's t-distribution "](https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Student_t_pdf.svg/488px-Student_t_pdf.svg.png)

Normal cumulative distribution function
![image Normal cumulative Student's t-distribution "](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Student_t_cdf.svg/488px-Student_t_cdf.svg.png)


![t-Tests for One Mean: Introduction](http://nikbearbrown.com/YouTube/MachineLearning/IMG/t_Tests_for_One_Mean_Introduction.png)

t-Tests for One Mean: Introduction [https://youtu.be/T9nI6vhTU1Y](https://youtu.be/T9nI6vhTU1Y)   

![t-Tests for One Mean: An Example](http://nikbearbrown.com/YouTube/MachineLearning/IMG/t_Tests_for_One_Mean_An_Example.png)

t-Tests for One Mean: An Example [https://youtu.be/kQ4xcx6N0o4](https://youtu.be/kQ4xcx6N0o4)  

![Hypothesis tests on one mean: t-test or z-test?](http://nikbearbrown.com/YouTube/MachineLearning/IMG/Hypothesis_tests_on_one_mean_t_test_or_z_test.png)

Hypothesis tests on one mean: t-test or z-test? [https://youtu.be/vw2IPZ2aD-c](https://youtu.be/vw2IPZ2aD-c)  


![Hypothesis testing and p-values](http://nikbearbrown.com/YouTube/MachineLearning/IMG/Hypothesis_testing_and_p-values.png)

Hypothesis testing and p-values [https://youtu.be/-FtlH4svqx4](https://youtu.be/-FtlH4svqx4)     

## Which of distributions below are significantly different?


```python
# Plot two normal distributions
domain = np.arange(-22, 33, 0.1)
values = stats.norm(3.3, 5.5).pdf(domain)
plt.plot(domain, values, color='r', linewidth=2)
plt.fill_between(domain, 0, values, color='#ffb6c1', alpha=0.3)
values = stats.norm(4.4, 6.6).pdf(domain)
plt.plot(domain, values, color='b', linewidth=2)
plt.fill_between(domain, 0, values, color='#89cff0', alpha=0.3)
plt.ylabel("Probability")
plt.title("Normal Distributions")
plt.show()
```


    
![png](output_6_0.png)
    



```python
# Plot two normal distributions
domain = np.arange(1, 15, 0.1)
values = stats.norm(5.5, 1.1).pdf(domain)
plt.plot(domain, values, color='r', linewidth=2)
plt.fill_between(domain, 0, values, color='#ffb6c1', alpha=0.3)
values = stats.norm(9.9, 1.1).pdf(domain)
plt.plot(domain, values, color='b', linewidth=2)
plt.fill_between(domain, 0, values, color='#89cff0', alpha=0.3)
plt.ylabel("Probability")
plt.title("Normal Distributions")
plt.show()
```


    
![png](output_7_0.png)
    


## Statistical hypothesis tests

There are a few good statistical tests for hypothesis testing:
* [ANOVA](https://en.wikipedia.org/wiki/Analysis_of_variance)
* [Welch's t-test](https://en.wikipedia.org/wiki/Welch's_t-test)
* [Mann-Whitney test](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test)

Each test makes various assumptions:

* ANOVA assumes normal distributions and equal variances in the two data sets
* The Welch t-test assumes normal distributions but not necessarily equal variances, and accounts for small sample sizes better
* The Mann-Whitney test assumes nothing about the distributions but requires at least 20 data points in each set, and produces a weaker p-value

Typically you need to choose the most appropriate test. Tests that make more assumptions are more discriminating (stronger p-values) but can be misleading on data sets that don't satisfy the assumptions.


All of these tests are available in the `scipy` library, a stats library for python:
* [ANOVA](http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.stats.f_oneway.html)
* [Welch's t-test](http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
* [Mann-Whitney](http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.stats.mannwhitneyu.html)


```python
# Generate some rnd_data_ and randomize

rnd_data_1 = []
rnd_data_1.extend(stats.norm(3.3, 5.5).rvs(3333))
np.random.shuffle(rnd_data_1)

rnd_data_2 = []
rnd_data_2.extend(stats.norm(4.4, 6.6).rvs(3333))
np.random.shuffle(rnd_data_2)

rnd_data_3 = []
rnd_data_3.extend(stats.norm(5.5, 1.1).rvs(3333))
np.random.shuffle(rnd_data_3)

rnd_data_4 = []
rnd_data_4.extend(stats.norm(9.9, 1.1).rvs(3333))
np.random.shuffle(rnd_data_4)

rnd_data_5 = []
rnd_data_5.extend(stats.norm(9.9, 1.1).rvs(3333))
np.random.shuffle(rnd_data_5)

# Make a rnd_data_ frame
rnd_data = pd.DataFrame()
rnd_data["A"] = rnd_data_1
rnd_data["B"] = rnd_data_2
rnd_data["C"] = rnd_data_3
rnd_data["D"] = rnd_data_4
rnd_data["E"] = rnd_data_5
rnd_data.head()
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
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.518942</td>
      <td>20.208505</td>
      <td>6.742951</td>
      <td>11.486656</td>
      <td>9.087031</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-3.801092</td>
      <td>-1.492341</td>
      <td>6.303298</td>
      <td>10.426167</td>
      <td>7.468687</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.900099</td>
      <td>4.029701</td>
      <td>8.146613</td>
      <td>9.993979</td>
      <td>10.697729</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.440442</td>
      <td>-3.941672</td>
      <td>3.993074</td>
      <td>9.848137</td>
      <td>9.638242</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.996488</td>
      <td>8.173902</td>
      <td>5.227747</td>
      <td>10.409755</td>
      <td>8.144193</td>
    </tr>
  </tbody>
</table>
</div>




```python
rnd_data.hist()
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7fe5846b42e8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe5845a89e8>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7fe5845dac50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe584590eb8>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7fe584550160>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe584760240>]],
          dtype=object)




    
![png](output_10_1.png)
    



```python
# To use ANOVA
statistic, pvalue = stats.f_oneway(rnd_data["A"], rnd_data["B"])
print (pvalue)
print (statistic)
```

    3.0228309569730286e-09
    35.26471029781106
    


```python
# To use ANOVA
statistic, pvalue = stats.f_oneway(rnd_data["C"], rnd_data["D"])
print (pvalue)
print (statistic)
```

    0.0
    26700.322335627967
    


```python
# To use ANOVA
statistic, pvalue = stats.f_oneway(rnd_data["A"], rnd_data["A"])
print (pvalue)
print (statistic)
```

    nan
    -1.1018629791790628e-31
    


```python
# To use ANOVA
statistic, pvalue = stats.f_oneway(rnd_data["D"], rnd_data["E"])
print (pvalue)
print (statistic)
```

    0.7241263781690648
    0.1245837877468245
    


```python
# to use Welch
statistic, pvalue = stats.ttest_ind(rnd_data["A"], rnd_data["B"])
print (pvalue)
print (statistic)
```

    3.02283095697572e-09
    -5.938409744856867
    


```python
# to use Welch
statistic, pvalue = stats.ttest_ind(rnd_data["C"], rnd_data["D"])
print (pvalue)
print (statistic)
```

    0.0
    -163.40233271170862
    


```python
# to use Welch
statistic, pvalue = stats.ttest_ind(rnd_data["B"], rnd_data["B"])
print (pvalue)
print (statistic)
```

    1.0
    0.0
    


```python
# to use Welch
statistic, pvalue = stats.ttest_ind(rnd_data["D"], rnd_data["E"])
print (pvalue)
print (statistic)
```

    0.7241263781686158
    0.35296428678663244
    


```python
# Use the Mann-Whitney test on our data
# look up the function in scipy from the link above
# stats.mannwhitneyu
statistic, pvalue = stats.mannwhitneyu(rnd_data["A"], rnd_data["B"])
print (pvalue)
print (statistic)
```

    6.148584981036516e-10
    5077079.0
    


```python
# Use the Mann-Whitney test on our data
# look up the function in scipy from the link above
# stats.mannwhitneyu
statistic, pvalue = stats.mannwhitneyu(rnd_data["C"], rnd_data["D"])
print (pvalue)
print (statistic)
```

    0.0
    24042.0
    


```python
# Use the Mann-Whitney test on our data
# look up the function in scipy from the link above
# stats.mannwhitneyu
statistic, pvalue = stats.mannwhitneyu(rnd_data["C"], rnd_data["C"])
print (pvalue)
print (statistic)
```

    0.49999746095635295
    5554444.5
    


```python
# Use the Mann-Whitney test on our data
# look up the function in scipy from the link above
# stats.mannwhitneyu
statistic, pvalue = stats.mannwhitneyu(rnd_data["D"], rnd_data["E"])
print (pvalue)
print (statistic)
```

    0.3045282129334781
    5514266.0
    


```python
mw = stats.mannwhitneyu(rnd_data["A"], rnd_data["B"])
print (mw)
print (mw.statistic)
```

    MannwhitneyuResult(statistic=5077079.0, pvalue=6.148584981036516e-10)
    5077079.0
    



## Inferential Statistical Tests

* Chi Square • compares observed frequencies to expected frequencies.   
* t-Test • looks at differences between two groups on some variable of interest.  
* Welch-Test • looks at differences between two groups on some variable of interest.
* Mann-Whitney test • looks at differences between two groups on some variable of interest.
* ANOVA • tests the significance of group differences between two or more groups. (Only determines that there is a difference between groups, but doesn’t tell which is different.)


##  One Way ANOVA Vs Two Way ANOVA

One way ANOVA takes only one factor (i.e. independent variable). Two way ANOVA assesses two factors concurrently.

|   | ONE WAY ANOVA    | TWO WAY ANOVA    |
|---|------------------|------------------|
|   | One way ANOVA is a hypothesis test.  | Two way ANOVA is a statistical technique assessing the interaction between factors. |
| Independent Variables  | One  | Two  |
|Number of Observations   | Need not to be same in each group. |  Need to be equal in each group.  |


Last update May 4, 2018
