# Probability Distributions

## Topics

* Probability
* Random variables
* Probability distributions
    * Uniform
    * Normal
    * Binomial
    * Poisson     
    * Fat Tailed
    
## Probability

* Probability is a measure of the likelihood of a random phenomenon or chance behavior.  Probability describes the long-term proportion with which a certain outcome will occur in situations with short-term uncertainty.
* Probability is expressed in numbers between 0 and 1.  Probability = 0 means the event never happens; probability = 1 means it always happens.
* The total probability of all possible event always sums to 1.

## Sample Space

* Coin Toss ={head,tail}
* Two coins S = {HH, HT, TH, TT}
* Inspecting a part ={good,bad}
* Rolling a die S ={1,2,3,4,5,6}

## Random Variables

In probability and statistics, a random variable,  or stochastic variable is a variable whose value is subject to variations due to chance (i.e. it can take on a range of values)


* Coin Toss ={head,tail}
* Rolling a die S ={1,2,3,4,5,6}

Discrete Random Variables

* Random variables (RVs) which may take on only a countable number of distinct values
E.g. the total number of tails X you get if you flip 100 coins
* X is a RV with arity k if it can take on exactly one value out of {x1, …, xk}
E.g. the possible values that X can take on are 0, 1, 2, …, 100

Continuous Random Variables

* Probability density function (pdf) instead of probability mass function (pmf)
* A pdf is any function f(x) that describes the probability density in terms of the input variable x.


## Probability distributions

* We use probability distributions because they model data in real world.
* They allow us to calculate what to expect and therefore understand what is unusual.
* They also provide insight in to the process in which real world data may have been generated.
* Many machine learning algorithms have assumptions based on certain probability distributions.

_Cumulative distribution function_

A probability distribution Pr on the real line is determined by the probability of a scalar random variable X being in a half-open interval (-$\infty$, x], the probability distribution is completely characterized by its cumulative distribution function:

$$
 F(x) = \Pr[X \leq x] \quad \forall \quad x \in R .
$$


## Uniform Distribution

$$
X \equiv U[a,b]
$$

$$
 f(x) = \frac{1}{b-a} \quad for \quad a \lt x \lt b
$$

$$
 f(x) = 0 \quad for \quad a \leq x  \quad or  \quad \geq b
$$

$$
 F(x) = \frac{x-a}{b-a} \quad for \quad a \leq x \lt b
$$

$$
F(x) = 0 \quad for \quad x  \lt a  \quad
 F(x) = 1 \quad for \quad x  \geq b
$$

![image Uniform Distribution"](http://54.198.163.24/YouTube/MachineLearning/M01/Uniform_Distribution_A.png)

_Continuous Uniform Distribution_

In probability theory and statistics, the continuous uniform distribution or rectangular distribution is a family of symmetric probability distributions such that for each member of the family, all intervals of the same length on the distribution's support are equally probable.

- from [Uniform distribution (continuous  Wikipedia)](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous))
    

![image continuous  Uniform Distribution"](https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Uniform_Distribution_PDF_SVG.svg/375px-Uniform_Distribution_PDF_SVG.svg.png)
![image continuous  Uniform Distribution"](https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Uniform_cdf.svg/375px-Uniform_cdf.svg.png)

_Discrete Uniform Distribution_

In probability theory and statistics, the discrete uniform distribution is a symmetric probability distribution whereby a finite number of values are equally likely to be observed; every one of n values has equal probability 1/n. Another way of saying "discrete uniform distribution" would be "a known, finite number of outcomes equally likely to happen".

- from [Uniform distribution (discrete)  Wikipedia)](https://en.wikipedia.org/wiki/Uniform_distribution_(discrete))
    

![image Uniform distribution (discrete) "](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Uniform_discrete_pmf_svg.svg/488px-Uniform_discrete_pmf_svg.svg.png)
![imageUniform distribution (discrete) "](https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/Dis_Uniform_distribution_CDF.svg/488px-Dis_Uniform_distribution_CDF.svg.png)



## Uniform Distribution in python


```python
%matplotlib inline
# %matplotlib inline is a magic function in IPython that displays images in the notebook
# Line magics are prefixed with the % character and work much like OS command-line calls
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.testing as tm
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Make plots larger
plt.rcParams['figure.figsize'] = (10, 6)
```


```python
#------------------------------------------------------------
# Define the distribution parameters to be plotted
W_values = [1.0, 3.0, 5.0]
linestyles = ['-', '--', ':']
mu = 0
x = np.linspace(-4, 4, 1000)

#------------------------------------------------------------
# plot the distributions
fig, ax = plt.subplots(figsize=(10, 5))

for W, ls in zip(W_values, linestyles):
    left = mu - 0.5 * W
    dist = stats.uniform(left, W)

    plt.plot(x, dist.pdf(x), ls=ls, c='black',
             label=r'$\mu=%i,\ W=%i$' % (mu, W))

plt.xlim(-4, 4)
plt.ylim(0, 1.2)

plt.xlabel('$x$')
plt.ylabel(r'$p(x|\mu, W)$')
plt.title('Uniform Distribution')

plt.legend()
plt.show()

# Adapted from http://www.astroml.org/book_figures/chapter3/fig_uniform_distribution.html

```


    
![png](output_4_0.png)
    


## Quiz Distribution of two dice

See if you can generate a distribution that models the output that would be generated by the sum of two dice.  Self-test homework.

## Normal Distribution

In probability theory, the normal (or Gaussian) distribution is a very common continuous probability distribution. The normal distribution is remarkably useful because of the central limit theorem. In its most general form, under mild conditions, it states that averages of random variables independently drawn from independent distributions are normally distributed. Physical quantities that are expected to be the sum of many independent processes (such as measurement errors) often have distributions that are nearly normal.

- from [Normal Distribution - Wikipedia)](https://en.wikipedia.org/wiki/Normal_distribution)
   

$$
X \sim \quad N(\mu, \sigma^2)
$$


$$
 f(x) = \frac{1}{\sigma \sqrt {2\pi }} e^{-\frac{( x - \mu)^2}{2\sigma^2}} \quad
$$


![image Normal Distribution"](http://54.198.163.24/YouTube/MachineLearning/M01/Normal_Distribution_A.png)


![image Normal Distribution  "](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/525px-Normal_Distribution_PDF.svg.png)

Normal cumulative distribution function
![image Normal cumulative distribution function "](https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Normal_Distribution_CDF.svg/525px-Normal_Distribution_CDF.svg.png)


_Properties of normal distribution_

- symmetrical, unimodal, and bell-shaped
- on average, the error component will equal zero, the error above and below the mean will cancel out
- Z-Score is a statistical measurement is (above/below) the mean of the data
- important characteristics about z scores:
  1. mean of z scores is 0
  2. standard deviation of a standardized variable is always 1
  3. the linear transformation does not change the _form_ of the distribution


The normal (or Gaussian) distribution was discovered in 1733 by Abraham de Moivre as an approximation to the binomial distribution when the number of trails is large.    

![image Abraham de Moivre "](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Abraham_de_moivre.jpg/300px-Abraham_de_moivre.jpg)

- from [Abraham de Moivre - Wikipedia)](https://en.wikipedia.org/wiki/Abraham_de_Moivre)

The Gaussian distribution was derived in 1809 by Carl Friedrich Gauss.    

![image Carl Friedrich Gauss "](https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Carl_Friedrich_Gauss.jpg/330px-Carl_Friedrich_Gauss.jpg)

- from [Carl Friedrich Gauss - Wikipedia)](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss)

Importance lies in the Central Limit Theorem, which states that the sum of a large number of independent random variables (binomial, Poisson, etc.) will approximate a normal distribution


## Central Limit Theorem

In probability theory, the central limit theorem (CLT) states that, given certain conditions, the arithmetic mean of a sufficiently large number of iterates of independent random variables, each with a well-defined expected value and well-defined variance, will be approximately normally distributed, regardless of the underlying distribution. The central limit theorem has a number of variants. In its common form, the random variables must be identically distributed.

- from [Central Limit Theorem - Wikipedia)](https://en.wikipedia.org/wiki/Central_limit_theorem)
   

The Central Limit Theorem tells us that when the sample size is large the average $\bar{Y}$ of a random sample follows a normal distribution centered at the population average $\mu_Y$ and with standard deviation equal to the population standard deviation $\sigma_Y$, divided by the square root of the sample size $N$.

This means that if we subtract a constant from a random variable, the mean of the new random variable shifts by that constant. If $X$ is a random variable with mean $\mu$ and $a$ is a constant, the mean of $X - a$ is $\mu-a$.

This property also holds for the spread, if $X$ is a random variable with mean $\mu$ and SD $\sigma$, and $a$ is a constant, then the mean and SD of $aX$ are $a \mu$ and $\|a\| \sigma$ respectively.
This implies that if we take many samples of size $N$ then the quantity

$$
\frac{\bar{Y} - \mu}{\sigma_Y/\sqrt{N}}
$$

is approximated with a normal distribution centered at 0 and with standard deviation 1.




```python

```


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


## Normal Distribution in python


```python
# Plot two normal distributions
domain = np.arange(-22, 22, 0.1)
values = stats.norm(3.3, 5.5).pdf(domain)
plt.plot(domain, values, color='r', linewidth=2)
plt.fill_between(domain, 0, values, color='#ffb6c1', alpha=0.3)
values = stats.norm(4.4, 2.3).pdf(domain)
plt.plot(domain, values, color='b', linewidth=2)
plt.ylabel("Probability")
plt.title("Two Normal Distributions")
plt.show()
```


    
![png](output_10_0.png)
    


## Binomial Distribution


$$
X \quad \sim \quad B(n, p)
$$


$$
 P(X=k) = \binom{n}{k} p^k (1-p)^{n-k} \quad k=1,2,...,n
$$

$$
    \binom{n}{k} = \frac{n!}{k!(n-k)!}
$$

_Binomial Distribution_

In probability theory and statistics, the binomial distribution with parameters n and p is the discrete probability distribution of the number of successes in a sequence of n independent yes/no experiments, each of which yields success with probability p. A success/failure experiment is also called a Bernoulli experiment or Bernoulli trial; when n = 1, the binomial distribution is a Bernoulli distribution.

- from [Binomial Distribution - Wikipedia](https://en.wikipedia.org/wiki/Binomial_distribution)


   
Binomial Distribution    
![image Binomial Distribution  "](https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/Binomial_distribution_pmf.svg/450px-Binomial_distribution_pmf.svg.png)

Binomial cumulative distribution function
![image Binomial cumulative distribution function "](https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Binomial_distribution_cdf.svg/450px-Binomial_distribution_cdf.svg.png)


* The data arise from a sequence of n independent trials.
* At each trial there are only two possible outcomes, conventionally called success and failure.
* The probability of success, p, is the same in each trial.
* The random variable of interest is the number of successes, X, in the n trials.
* The assumptions of independence and constant p are important. If they are invalid,  so is the binomial distribution

_Bernoulli Random Variables_

* Imagine a simple trial with only two possible outcomes
    * Success (S) with probabilty p.
    * Failure (F) with probabilty 1-p.

* Examples
    * Toss of a coin (heads or tails)
    * Gender of a newborn (male or female)



## Binomial Distribution in python


```python
#------------------------------------------------------------
# Define the distribution parameters to be plotted
n_values = [20, 20, 40]
b_values = [0.2, 0.6, 0.6]
linestyles = ['-', '--', ':']
x = np.arange(-1, 200)

#------------------------------------------------------------
# plot the distributions

for (n, b, ls) in zip(n_values, b_values, linestyles):
    # create a binomial distribution
    dist = stats.binom(n, b)

    plt.plot(x, dist.pmf(x), ls=ls, c='black',
             label=r'$b=%.1f,\ n=%i$' % (b, n), linestyle='steps-mid')

plt.xlim(-0.5, 35)
plt.ylim(0, 0.25)

plt.xlabel('$x$')
plt.ylabel(r'$p(x|b, n)$')
plt.title('Binomial Distribution')

plt.legend()
plt.show()

# Adapted from http://www.astroml.org/book_figures/chapter3/fig_binomial_distribution.html
```


    
![png](output_13_0.png)
    



```python
fair_coin_flips = stats.binom.rvs(n=33,        # Number of flips per trial
                                  p=0.4,       # Success probability
                                  size=1000)  # Number of trials
pd.DataFrame(fair_coin_flips).hist(range=(-0.5,10.5), bins=11)
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7f77785ae748>]],
          dtype=object)




    
![png](output_14_1.png)
    



```python
plt.fill_between(x=np.arange(-4,-1,0.01),
                 y1= stats.norm.pdf(np.arange(-4,-1,0.01)) ,
                 facecolor='red',
                 alpha=0.35)

plt.fill_between(x=np.arange(1,4,0.01),
                 y1= stats.norm.pdf(np.arange(1,4,0.01)) ,
                 facecolor='red',
                 alpha=0.35)

plt.fill_between(x=np.arange(-1,1,0.01),
                 y1= stats.norm.pdf(np.arange(-1,1,0.01)) ,
                 facecolor='blue',
                 alpha=0.35)
```




    <matplotlib.collections.PolyCollection at 0x7f777853b828>




    
![png](output_15_1.png)
    


## Poisson Distribution

$X$ expresses the number of "rare" events

$$
X \quad \sim P( \lambda )\quad \lambda \gt 0
$$

$$
    P(X = x) = \frac{ \mathrm{e}^{- \lambda } \lambda^x }{x!}  \quad x=1,2,...,n
$$


_Poisson Distribution_

In probability theory and statistics, the Poisson distribution, named after French mathematician Siméon Denis Poisson, is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time and/or space if these events occur with a constant rate per time unit and independently of the time since the last event. The Poisson distribution can also be used for the number of events in other specified intervals such as distance, area or volume.

For instance, an individual keeping track of the amount of mail they receive each day may notice that they receive an average number of 4 letters per day. If receiving any particular piece of mail doesn't affect the arrival times of future pieces of mail, i.e., if pieces of mail from a wide range of sources arrive independently of one another, then a reasonable assumption is that the number of pieces of mail received per day obeys a Poisson distribution. Other examples that may follow a Poisson: the number of phone calls received by a call center per hour, the number of decay events per second from a radioactive source, or the number of taxis passing a particular street corner per hour.

The Poisson distribution gives us a probability mass for discrete natural numbers *k* given some mean value &lambda;. Knowing that, on average, &lambda; discrete events occur over some time period, the Poisson distribution gives us the probability of seeing exactly *k* events in that time period.

For example, if a call center gets, on average, 100 customers per day, the Poisson distribution can tell us the probability of getting exactly 150 customers today.

*k* &isin; **N** (i.e. is a natural number) because, on any particular day, you can't have a fraction of a phone call. The probability of any non-integer number of people calling in is zero. E.g., P(150.5) = 0.

&lambda; &isin; **R** (i.e. is a real number) because, even though any *particular* day must have an integer number of people, the *mean* number of people taken over many days can be fractional (and usually is). It's why the "average" number of phone calls per day could be 3.5 even though half a phone call won't occur.


- from [Poisson Distribution - Wikipedia)](https://en.wikipedia.org/wiki/Poisson_distribution)
   
Poisson Distribution    
![image Poisson Distribution  "](https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Poisson_pmf.svg/488px-Poisson_pmf.svg.png)

Poisson cumulative distribution function
![image Poisson cumulative distribution function "](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/Poisson_cdf.svg/488px-Poisson_cdf.svg.png)


_Properties of Poisson distribution_

* The mean number of successes from n trials is µ = np
* If we substitute µ/n for p, and let n tend to infinity, the binomial distribution becomes the Poisson distribution.
* Poisson distributions are often used to describe the number of occurrences of a ‘rare’ event. For example
    * The number of storms in a season
    * The number of occasions in a season when river levels exceed a certain value
* The main assumptions are that events occur
    * at random (the occurrence of an event doesn’t change the probability of  it happening again)
    *  at a constant rate
* Poisson distributions also arise as approximations  to  binomials when n is large and p is small.
* When there is a large number of trials, but a very small probability of success, binomial calculation becomes impractical



## Poisson Distribution in python


```python
# Generate poisson counts
arrival_rate_1 = stats.poisson.rvs(size=10000,  # Generate Poisson data
                                   mu=1 )       # Average arrival time 1


# Plot histogram
pd.DataFrame(arrival_rate_1).hist(range=(-0.5,max(arrival_rate_1)+0.5)
                                    , bins=max(arrival_rate_1)+1)
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7f77785bfe80>]],
          dtype=object)




    
![png](output_18_1.png)
    



```python
arrival_rate_10 = stats.poisson.rvs(size=10000,  # Generate Poisson data
                                   mu=10 )       # Average arrival time 10

# Plot histogram
pd.DataFrame(arrival_rate_10).hist(range=(-0.5,max(arrival_rate_10)+0.5)
                                    , bins=max(arrival_rate_10)+1)
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7f777a048668>]],
          dtype=object)




    
![png](output_19_1.png)
    


## Poisson and Binomial Distributions

The binomial distribution is usually shown with a fixed n, with different values of p that will affect the k successes from the fixed n trails. This supposes we know the number of trails beforehand.  We can graph the binomial distribution as a set of curves with a fixed n, and varying probabilities of the probability of success, p, below.



## What if we knew the rate but not the probability, p, or the number of trails, n?

But what if we were to invert the problem? What if we knew only the number of heads we observed, but not the total number of flips? If we have a known expected number of heads but an unknown number of flips, then we don't really know the true probability for each individual head. Rather we know that, on average, p=mean(k)/n. However if we were to plot these all on the same graph in the vicinity of the same k, we can make them all have a convergent shape around mean(k) because, no matter how much we increase n, we decrease p proportionally so that, for all n, the peak stays at mean(k).

## Deriving the Poisson Distribution from the Binomial Distribution

Let’s make this a little more formal. The binomial distribution works when we have a fixed number of events n, each with a constant probability of success p. In the Poisson Distribution, we don't know the number of trials that will happen. Instead, we only know the average number of successes per time period, the rate $\lambda$. So we know the rate of successes per day, or per minute but not the number of trials n or the probability of success p that was used to estimate to that rate.

If n is the number of trails in our time period, then np is the success rate or $\lambda$, that is, $\lambda$ = np. Solving for p, we get:

$$
p=\frac{\lambda}{n} \quad(1)
$$
Since the Binomial distribution is defined as below
$$
 P(X=k) = \binom{n}{k} p^k (1-p)^{n-k} \quad k=1,2,...,n
\quad  (2)
$$
or equivelently
$$
 P(X=k) = \frac{n!}{k!(n-k)!} p^k (1-p)^{n-k} \quad k=1,2,...,n
\quad  (3)
$$
By substituting the above p from (1) into the binomial distribution (3)
 $$
 P(X=k) = \frac{n!}{k!(n-k)!} {\frac{\lambda!}{n}}^k (1-{\frac{\lambda!}{n} })^{n-k}  \quad  (4)
 $$

         

For n large and p small:

$$
    P(X = k) \equiv \frac{ \mathrm{e}^{- \lambda } \lambda^k }{k!}  \quad k=1,2,...,n\quad  (5)
$$
                                        

Which is the probability mass function for the Poisson distribution.


## Fat-Tailed Distribution


In probability theory, the Fat-Tailed (or Gaussian) distribution is a very common continuous probability distribution. The Fat-Tailed distribution is remarkably useful because of the central limit theorem. In its most general form, under mild conditions, it states that averages of random variables independently drawn from independent distributions are Fat-Tailedly distributed. Physical quantities that are expected to be the sum of many independent processes (such as measurement errors) often have distributions that are nearly Fat-Tailed.

- from [Fat-Tailed Distribution - Wikipedia)](https://en.wikipedia.org/wiki/Fat-Tailed_distribution)
   

_Properties of Fat-Tailed distribution_

* Power law distributions:
    * for variables assuming integer values > 0
    * Prob [X=k] ~ Ck-α
    * typically 0 < alpha < 2; smaller a gives heavier tail  
* For binomial, normal, and Poisson distributions the tail probabilities approach 0 exponentially fast
* What kind of phenomena does this distribution model?
* What kind of process would generate it?



## Cauchy Distribution

An example of a Fat-tailed distribution is the Cauchy distribution.

 The Cauchy distribution, named after Augustin Cauchy, is a continuous probability distribution. It is also known, especially among physicists, as the Lorentz distribution (after Hendrik Lorentz), Cauchy–Lorentz distribution, Lorentz(ian) function, or Breit–Wigner distribution. The simplest Cauchy distribution is called the standard Cauchy distribution. It is the distribution of a random variable that is the ratio of two independent standard normal variables and has the probability density function

The Cauchy distribution is often used in statistics as the canonical example of a "pathological" distribution since both its mean and its variance are undefined. (But see the section Explanation of undefined moments below.) The Cauchy distribution does not have finite moments of order greater than or equal to one; only fractional absolute moments exist.[1] The Cauchy distribution has no moment generating function.

- from [Cauchy Distribution - Wikipedia)](https://en.wikipedia.org/wiki/Cauchy_distribution)
   
Cauchy Distribution    
![image Cauchy Distribution  "](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Cauchy_pdf.svg/450px-Cauchy_pdf.svg.png)

Cauchy cumulative distribution function
![image Cauchy cumulative distribution function "](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Cauchy_cdf.svg/450px-Cauchy_cdf.svg.png)






## Cauchy Distribution in python


```python
# Define the distribution parameters to be plotted
gamma_values = [0.5, 1.0, 2.0]
linestyles = ['-', '--', ':']
mu = 0
x = np.linspace(-10, 10, 1000)

#------------------------------------------------------------
# plot the distributions

for gamma, ls in zip(gamma_values, linestyles):
    dist = stats.cauchy(mu, gamma)

    plt.plot(x, dist.pdf(x), ls=ls, color='black',
             label=r'$\mu=%i,\ \gamma=%.1f$' % (mu, gamma))

plt.xlim(-4.5, 4.5)
plt.ylim(0, 0.65)

plt.xlabel('$x$')
plt.ylabel(r'$p(x|\mu,\gamma)$')
plt.title('Cauchy Distribution')

plt.legend()
plt.show()

# From http://www.astroml.org/book_figures/chapter3/fig_cauchy_distribution.html
```


    
![png](output_26_0.png)
    



```python
n=50
def random_distributions(n=50):
  mu, sigma, p = 5, 2*np.sqrt(2), 0.3# mean, standard deviation, probabilty of success
  shape, scale = 2.5, 2. # mean=5, std=2*sqrt(2)
  normal_dist = np.random.normal(mu, sigma, n)
  lognormal_dist = np.random.lognormal(mu, sigma, n)
  lognormal_dist = np.random.lognormal(np.log2(mu), np.log2(sigma), n)
  pareto_dist = np.random.pareto(mu, n)
  uniform_dist= np.random.uniform(np.amin(normal_dist),np.amax(normal_dist),n)
  binomial_dist= np.random.binomial(n, p,n)
  gamma_dist= np.random.gamma(shape, scale, n)
  poisson_dist= np.random.poisson((n*0.05), n)
  df = pd.DataFrame({'Normal' : normal_dist, 'Lognormal' : lognormal_dist, 'Pareto' : pareto_dist,'Gamma' : gamma_dist, 'Poisson' : poisson_dist, 'Binomial' : binomial_dist, 'Uniform' : uniform_dist})
  return df
```


```python
df=random_distributions(n=50)
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
      <th>Normal</th>
      <th>Lognormal</th>
      <th>Pareto</th>
      <th>Gamma</th>
      <th>Poisson</th>
      <th>Binomial</th>
      <th>Uniform</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.317971</td>
      <td>6.519577</td>
      <td>0.390445</td>
      <td>1.176659</td>
      <td>2</td>
      <td>17</td>
      <td>9.569800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.251573</td>
      <td>13.071381</td>
      <td>0.221789</td>
      <td>7.314131</td>
      <td>1</td>
      <td>17</td>
      <td>1.465310</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.851357</td>
      <td>0.570996</td>
      <td>0.067470</td>
      <td>1.136784</td>
      <td>4</td>
      <td>13</td>
      <td>1.118340</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.615446</td>
      <td>5.817909</td>
      <td>0.634731</td>
      <td>2.368797</td>
      <td>2</td>
      <td>15</td>
      <td>2.513710</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.639505</td>
      <td>2.933502</td>
      <td>0.038660</td>
      <td>3.452008</td>
      <td>1</td>
      <td>17</td>
      <td>9.633049</td>
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
      <th>Normal</th>
      <th>Lognormal</th>
      <th>Pareto</th>
      <th>Gamma</th>
      <th>Poisson</th>
      <th>Binomial</th>
      <th>Uniform</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.048293</td>
      <td>30.370469</td>
      <td>0.246787</td>
      <td>4.988022</td>
      <td>250.581600</td>
      <td>1499.234600</td>
      <td>5.300932</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.833359</td>
      <td>75.863151</td>
      <td>0.299481</td>
      <td>3.156553</td>
      <td>15.884641</td>
      <td>32.131462</td>
      <td>5.882226</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-4.834438</td>
      <td>0.018371</td>
      <td>0.000067</td>
      <td>0.025656</td>
      <td>197.000000</td>
      <td>1387.000000</td>
      <td>-4.834025</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.126114</td>
      <td>3.660994</td>
      <td>0.059181</td>
      <td>2.694034</td>
      <td>240.000000</td>
      <td>1478.000000</td>
      <td>0.227947</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.067160</td>
      <td>10.275546</td>
      <td>0.150687</td>
      <td>4.328851</td>
      <td>250.000000</td>
      <td>1499.000000</td>
      <td>5.311802</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.954635</td>
      <td>27.216179</td>
      <td>0.317047</td>
      <td>6.603777</td>
      <td>261.000000</td>
      <td>1521.000000</td>
      <td>10.507013</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15.476859</td>
      <td>2211.604436</td>
      <td>5.162607</td>
      <td>26.677119</td>
      <td>314.000000</td>
      <td>1610.000000</td>
      <td>15.462169</td>
    </tr>
  </tbody>
</table>
</div>




```python
print (df.columns.values)
```

    ['Normal' 'Lognormal' 'Pareto' 'Gamma' 'Poisson' 'Binomial' 'Uniform']
    




```python
sns.distplot(df['Normal'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f77780de7f0>




    
![png](output_32_1.png)
    



```python
sns.distplot(df['Lognormal'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f7778033710>




    
![png](output_33_1.png)
    



```python
sns.distplot(df['Gamma'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f7777f329e8>




    
![png](output_34_1.png)
    



```python
sns.distplot(df['Poisson'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f77787ef5c0>




    
![png](output_35_1.png)
    



```python
sns.distplot(df['Binomial'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f777a0b12b0>




    
![png](output_36_1.png)
    



```python
sns.distplot(df['Uniform'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f7779b58b70>




    
![png](output_37_1.png)
    



```python
def qqplot_stats(obs, c):
    z = (obs-np.mean(obs))/np.std(obs)
    stats.probplot(z, dist="norm", plot=plt)
    plt.title("Normal Q-Q plot for " + c)
    plt.show()
```


```python
def qqplot_df(df):
    for col in list(df.columns.values):
      qqplot_stats(df[col], col)
qqplot_df(df)
```


    
![png](output_39_0.png)
    



    
![png](output_39_1.png)
    



    
![png](output_39_2.png)
    



    
![png](output_39_3.png)
    



    
![png](output_39_4.png)
    



    
![png](output_39_5.png)
    



    
![png](output_39_6.png)
    


## Statistical tests for normality (e.g. Shapiro-Wilk test,  Anderson-Darling test, scipy.stats.normaltest, etc.)


```python
def normality_stats(df):
    s={}
    for col in list(df.columns.values):
      s[col]={}
    for col in list(df.columns.values):
      s[col].update({'shapiro':stats.shapiro(df[col])})
      s[col].update({'anderson':stats.anderson(df[col], dist='norm')})
      s[col].update({'normaltest':stats.normaltest(df[col])})
    return s
```

## Shapiro-Wilk test
scipy.stats.shapiro [https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)

scipy.stats.shapiro
scipy.stats.shapiro(x, a=None, reta=False)[source]
Perform the Shapiro-Wilk test for normality.

The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a normal distribution.

Parameters:
x : array_like
Array of sample data.
a : array_like, optional
Array of internal parameters used in the calculation. If these are not given, they will be computed internally. If x has length n, then a must have length n/2.
reta : bool, optional
Whether or not to return the internally computed a values. The default is False.
Returns:
W : float
The test statistic.
p-value : float
The p-value for the hypothesis test.
a : array_like, optional
If reta is True, then these are the internally computed “a” values that may be passed into this function on future calls.

### Anderson-Darling test

scipy.stats.anderson [https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html)

scipy.stats.anderson(x, dist='norm')
Anderson-Darling test for data coming from a particular distribution

The Anderson-Darling test is a modification of the Kolmogorov- Smirnov test kstest for the null hypothesis that a sample is drawn from a population that follows a particular distribution. For the Anderson-Darling test, the critical values depend on which distribution is being tested against. This function works for normal, exponential, logistic, or Gumbel (Extreme Value Type I) distributions.

Parameters:
x : array_like
array of sample data
dist : {‘norm’,’expon’,’logistic’,’gumbel’,’gumbel_l’, gumbel_r’,
‘extreme1’}, optional the type of distribution to test against. The default is ‘norm’ and ‘extreme1’, ‘gumbel_l’ and ‘gumbel’ are synonyms.
Returns:
statistic : float
The Anderson-Darling test statistic
critical_values : list
The critical values for this distribution
significance_level : list
The significance levels for the corresponding critical values in percents. The function returns critical values for a differing set of significance levels depending on the distribution that is being tested against.

Note: The critical values are for a given significance level. When we want a smaller significance level, then we have to increase the critical values, assuming we are in the right, upper tail of the distribution.

### scipy.stats.normaltest

scipy.stats.normaltest [https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.normaltest.html](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.normaltest.html)
scipy.stats.normaltest(a, axis=0)
Tests whether a sample differs from a normal distribution.

This function tests the null hypothesis that a sample comes from a normal distribution. It is based on D’Agostino and Pearson’s [R251], [R252] test that combines skew and kurtosis to produce an omnibus test of normality.

Parameters:
a : array_like
The array containing the data to be tested.
axis : int or None
If None, the array is treated as a single data set, regardless of its shape. Otherwise, each 1-d array along axis axis is tested.
Returns:
k2 : float or array
s^2 + k^2, where s is the z-score returned by skewtest and k is the z-score returned by kurtosistest.
p-value : float or array
A 2-sided chi squared probability for the hypothesis test.


```python
norm_stats=normality_stats(df)
print (norm_stats)
```

    {'Normal': {'shapiro': (0.9702109098434448, 0.23605886101722717), 'anderson': AndersonResult(statistic=0.48083144977227477, critical_values=array([0.538, 0.613, 0.736, 0.858, 1.021]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ])), 'normaltest': NormaltestResult(statistic=1.047437935561788, pvalue=0.592313651439341)}, 'Lognormal': {'shapiro': (0.3546706438064575, 1.6915458020839297e-13), 'anderson': AndersonResult(statistic=12.167499087660822, critical_values=array([0.538, 0.613, 0.736, 0.858, 1.021]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ])), 'normaltest': NormaltestResult(statistic=85.13570673442072, pvalue=3.258488376580433e-19)}, 'Pareto': {'shapiro': (0.7625029683113098, 1.3276107324600162e-07), 'anderson': AndersonResult(statistic=4.166789341991091, critical_values=array([0.538, 0.613, 0.736, 0.858, 1.021]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ])), 'normaltest': NormaltestResult(statistic=25.69587562773078, pvalue=2.631549257633745e-06)}, 'Gamma': {'shapiro': (0.9117493033409119, 0.0012020422145724297), 'anderson': AndersonResult(statistic=1.2009411511148116, critical_values=array([0.538, 0.613, 0.736, 0.858, 1.021]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ])), 'normaltest': NormaltestResult(statistic=13.252346714577229, pvalue=0.0013252245565147113)}, 'Poisson': {'shapiro': (0.888938307762146, 0.0002113741938956082), 'anderson': AndersonResult(statistic=2.0576145876127256, critical_values=array([0.538, 0.613, 0.736, 0.858, 1.021]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ])), 'normaltest': NormaltestResult(statistic=6.197386714497713, pvalue=0.04510810408066107)}, 'Binomial': {'shapiro': (0.9712061882019043, 0.25895628333091736), 'anderson': AndersonResult(statistic=0.518999526034122, critical_values=array([0.538, 0.613, 0.736, 0.858, 1.021]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ])), 'normaltest': NormaltestResult(statistic=0.5307292184433161, pvalue=0.7669262706015215)}, 'Uniform': {'shapiro': (0.9445751309394836, 0.020518384873867035), 'anderson': AndersonResult(statistic=0.7313725429513624, critical_values=array([0.538, 0.613, 0.736, 0.858, 1.021]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ])), 'normaltest': NormaltestResult(statistic=13.040712958177991, pvalue=0.001473143863758758)}}
    


```python
df=random_distributions(n=500)
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
      <th>Normal</th>
      <th>Lognormal</th>
      <th>Pareto</th>
      <th>Gamma</th>
      <th>Poisson</th>
      <th>Binomial</th>
      <th>Uniform</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.055723</td>
      <td>12.758962</td>
      <td>0.881737</td>
      <td>1.874738</td>
      <td>30</td>
      <td>149</td>
      <td>0.341570</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.164571</td>
      <td>41.822388</td>
      <td>0.418558</td>
      <td>5.401163</td>
      <td>21</td>
      <td>138</td>
      <td>4.248585</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.422250</td>
      <td>5.254390</td>
      <td>0.100245</td>
      <td>3.792392</td>
      <td>28</td>
      <td>156</td>
      <td>-1.967696</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.468742</td>
      <td>60.234441</td>
      <td>0.548336</td>
      <td>1.518464</td>
      <td>26</td>
      <td>154</td>
      <td>2.142517</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.132351</td>
      <td>99.506178</td>
      <td>0.025369</td>
      <td>3.156522</td>
      <td>25</td>
      <td>162</td>
      <td>11.409817</td>
    </tr>
  </tbody>
</table>
</div>




```python
show_distributions(df)
qqplot_df(df)
```


    
![png](output_45_0.png)
    



    
![png](output_45_1.png)
    



    
![png](output_45_2.png)
    



    
![png](output_45_3.png)
    



    
![png](output_45_4.png)
    



    
![png](output_45_5.png)
    



    
![png](output_45_6.png)
    



```python
norm_stats=normality_stats(df)
print (norm_stats)
```

    {'Normal': {'shapiro': (0.9968737959861755, 0.4518488645553589), 'anderson': AndersonResult(statistic=0.23337016448982695, critical_values=array([0.571, 0.651, 0.781, 0.911, 1.083]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ])), 'normaltest': NormaltestResult(statistic=1.9717283634442144, pvalue=0.37311664698634833)}, 'Lognormal': {'shapiro': (0.37177103757858276, 3.3722906154254955e-38), 'anderson': AndersonResult(statistic=95.93512796242203, critical_values=array([0.571, 0.651, 0.781, 0.911, 1.083]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ])), 'normaltest': NormaltestResult(statistic=631.9117351507322, pvalue=6.054944783539203e-138)}, 'Pareto': {'shapiro': (0.6842437982559204, 1.7188155886311094e-29), 'anderson': AndersonResult(statistic=40.82409926781963, critical_values=array([0.571, 0.651, 0.781, 0.911, 1.083]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ])), 'normaltest': NormaltestResult(statistic=383.9713051507407, pvalue=4.184952007482177e-84)}, 'Gamma': {'shapiro': (0.9142821431159973, 3.2193514745517174e-16), 'anderson': AndersonResult(statistic=10.381010527113176, critical_values=array([0.571, 0.651, 0.781, 0.911, 1.083]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ])), 'normaltest': NormaltestResult(statistic=107.52108991857598, pvalue=4.488404406536119e-24)}, 'Poisson': {'shapiro': (0.9880919456481934, 0.00042510731145739555), 'anderson': AndersonResult(statistic=2.238733517790763, critical_values=array([0.571, 0.651, 0.781, 0.911, 1.083]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ])), 'normaltest': NormaltestResult(statistic=6.997592167615503, pvalue=0.03023376043422721)}, 'Binomial': {'shapiro': (0.9974591732025146, 0.6470588445663452), 'anderson': AndersonResult(statistic=0.44142954293454295, critical_values=array([0.571, 0.651, 0.781, 0.911, 1.083]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ])), 'normaltest': NormaltestResult(statistic=0.3611582708462676, pvalue=0.8347866168908302)}, 'Uniform': {'shapiro': (0.945564329624176, 1.381570747562011e-12), 'anderson': AndersonResult(statistic=7.040820053287803, critical_values=array([0.571, 0.651, 0.781, 0.911, 1.083]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ])), 'normaltest': NormaltestResult(statistic=534.6553227483895, pvalue=7.962909961540458e-117)}}
    


```python
df=random_distributions(n=5000)
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
      <th>Normal</th>
      <th>Lognormal</th>
      <th>Pareto</th>
      <th>Gamma</th>
      <th>Poisson</th>
      <th>Binomial</th>
      <th>Uniform</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.594114</td>
      <td>3.679353</td>
      <td>0.102433</td>
      <td>7.623831</td>
      <td>274</td>
      <td>1506</td>
      <td>1.078662</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.338785</td>
      <td>10.012055</td>
      <td>0.388363</td>
      <td>16.299674</td>
      <td>254</td>
      <td>1490</td>
      <td>3.553364</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.254345</td>
      <td>158.817914</td>
      <td>0.293478</td>
      <td>4.961992</td>
      <td>265</td>
      <td>1448</td>
      <td>10.161228</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.883146</td>
      <td>8.994029</td>
      <td>0.106616</td>
      <td>10.412256</td>
      <td>240</td>
      <td>1484</td>
      <td>6.528808</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.491562</td>
      <td>21.266680</td>
      <td>0.016310</td>
      <td>3.773419</td>
      <td>274</td>
      <td>1505</td>
      <td>-4.296865</td>
    </tr>
  </tbody>
</table>
</div>




```python
show_distributions(df)
qqplot_df(df)
```


    
![png](output_48_0.png)
    



    
![png](output_48_1.png)
    



    
![png](output_48_2.png)
    



    
![png](output_48_3.png)
    



    
![png](output_48_4.png)
    



    
![png](output_48_5.png)
    



    
![png](output_48_6.png)
    



```python
norm_stats=normality_stats(df)
print (norm_stats)
```

    {'Normal': {'shapiro': (0.9997493624687195, 0.8450976014137268), 'anderson': AndersonResult(statistic=0.26558287310399464, critical_values=array([0.576, 0.655, 0.786, 0.917, 1.091]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ])), 'normaltest': NormaltestResult(statistic=0.15398937564649992, pvalue=0.9258947721101127)}, 'Lognormal': {'shapiro': (0.34310734272003174, 0.0), 'anderson': AndersonResult(statistic=908.766116944762, critical_values=array([0.576, 0.655, 0.786, 0.917, 1.091]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ])), 'normaltest': NormaltestResult(statistic=8221.62690570761, pvalue=0.0)}, 'Pareto': {'shapiro': (0.7053076028823853, 0.0), 'anderson': AndersonResult(statistic=360.9078757297484, critical_values=array([0.576, 0.655, 0.786, 0.917, 1.091]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ])), 'normaltest': NormaltestResult(statistic=3911.856157100812, pvalue=0.0)}, 'Gamma': {'shapiro': (0.9151313304901123, 0.0), 'anderson': AndersonResult(statistic=89.88256526769692, critical_values=array([0.576, 0.655, 0.786, 0.917, 1.091]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ])), 'normaltest': NormaltestResult(statistic=1207.3011130710552, pvalue=6.884847083960581e-263)}, 'Poisson': {'shapiro': (0.9987745881080627, 0.000819850480183959), 'anderson': AndersonResult(statistic=1.726435529135415, critical_values=array([0.576, 0.655, 0.786, 0.917, 1.091]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ])), 'normaltest': NormaltestResult(statistic=11.536052362116626, pvalue=0.0031259214318674813)}, 'Binomial': {'shapiro': (0.9996830821037292, 0.6552309393882751), 'anderson': AndersonResult(statistic=0.38866632169811055, critical_values=array([0.576, 0.655, 0.786, 0.917, 1.091]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ])), 'normaltest': NormaltestResult(statistic=0.16318287010224553, pvalue=0.9216484350293355)}, 'Uniform': {'shapiro': (0.9532049298286438, 2.7309304894444857e-37), 'anderson': AndersonResult(statistic=58.803017787221506, critical_values=array([0.576, 0.655, 0.786, 0.917, 1.091]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ])), 'normaltest': NormaltestResult(statistic=4930.408377935504, pvalue=0.0)}}
    

Last update September 5, 2017
