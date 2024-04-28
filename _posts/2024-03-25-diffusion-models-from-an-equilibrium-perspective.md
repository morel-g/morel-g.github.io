---
title: Diffusion models from an equilibrium perspective
author: Guillaume Morel
date: 25 march 2024
lang: en
---

# Diffusion models from an equilibrium perspective
 
Score based [[1](#ref1),[2](#ref2)] and diffusion models [[3](#ref3),[4](#ref)] are a class of generative models that have recently gained popularity in the field of artificial intelligence for their ability to generate high-quality, detailed images, audio, and other types of data.  Score-based and diffusion models, despite adopting distinct perspectives, essentially represent two facets of the same underlying methodology. These models operate by transforming a random distribution of noise into a structured pattern or image through a process that gradually remove the noise of an image. 

<div style="display: flex; flex-direction: row;">
    <img src="/assets/images/diffusion/face_0.gif" style="width: 33%;" />
    <img src="/assets/images/diffusion/face_1.gif" style="width: 33%;" />
    <img src="/assets/images/diffusion/face_2.gif" style="width: 33%;" />
</div>

There are already some great online posts about diffusion models such as Lilian Weng post [[5](#ref5)] which offers a detail mathematical view from the 'diffusion' perspective or Yang Song post [[6](#ref6)] highlighting the score-based approach. These posts however do not focus on the complete physical derivation and perspective behind these models which will be the goal of this presentation. 

Indeed, even if the intuition behind diffusion models came initially from Thermodynamics equilibrium [[3](#ref3)], this foundational aspect has since been quietly set aside. Here we propose to derive and explain score-based / diffusion models (we will call them diffusion models for short) from an equilibrium perspective.

## General idea behind the generative process of continuous data

In machine learning, data is frequently represented as an unknown high dimensional probability distribution. The core of generative modeling is its ability to synthesize new samples from this complex distribution. A crucial challenge is to design methods that effectively scales to extremely high dimensions $d$. For instance in the case of an image dataset, $d$ is typically the number of pixels on an image.

  A common approach is to learn a transformation $F$ which changes the initial distribution $p_{0}$ into a simple known distribution often chosen to be Gaussian. Once the model has been adequately trained the learned neural network is utilized to map the high dimensional samples from the Gaussian distribution back to the original data distribution.


![Continuous generation process](/assets/images/diffusion/gen_basic.png)
**Figure 1:** *Transformation of a Gaussian distribution into an unknown distribution via a function $F^{-1}$. For machine learning applications $F^{-1}$ is typically learned through a neural network.*

For diffusion models, the score-based approach [[1](#ref1), [2](#ref2), [3](#ref3)] makes a direct link between the function $F$ and the Fokker-Planck equation.

## The Fokker-Planck equation and its equilibrium

In physics, the Fokker-Planck equation provides insight into the movements of tiny particles as they are influenced by unpredictable collisions among themselves. In this context, the equation captures two fundamental processes: drift, which represents the systematic motion of particles under deterministic forces, and diffusion, which accounts for the collisions.

<div style="display: flex; flex-direction: row;">
    <img src="/assets/images/diffusion/moons_forward.gif" style="width: 50%;" />
    <img src="/assets/images/diffusion/swissroll_forward.gif" style="width: 50%;" />
</div>

**Figure 2:** *Application of the Fokker-Planck equation (3)<!--ref-->  to two given densities. The distribution converges toward a Normal distribution exponentially fast.*

The Fokker-Planck equation under its more general form can be written

$$\partial_t p(x,t) = \nabla \cdot (g(x, t) p(x,t))+ h(x, t) \Delta p (x,t),  \tag{1}$$

 where $x \in \mathbb{R}^d$, $g$ is the drift term and $h$ a function modeling the diffusion coefficient. For machine learning applications we usually operate in very high dimension hence computing the full distribution $p$ at once is not tractable. One of the crucial property of equation (1)<!--ref--> is the possibility to adopt a particles based view 

$$
dx = -g(x) dt + \sqrt{2 h(x) dt} z, \quad z \sim \mathcal{N}(0, I).  \tag{2}
$$

This formulation allows for the simulation of independent particles, as described in (1)<!--ref-->, without the need to consider the entire density at once. This approach offers a significant advantage, particularly for machine learning applications, due to its effectiveness in high-dimensional spaces. 

From now on, we will focus on a specific form of the Fokker-Planck equation, which we'll simply refer to as the Fokker-Planck equation. That is equation (1)<!--ref--> with $g(x, t)=x$ and $h(x, t)=1$

$$\partial_t p(x,t) = \nabla \cdot (x p(x,t))+ \Delta p (x,t). \tag{3}$$

The fundamental property of this equation, on which diffusion models are based on, is that for any initial data it will always converge toward a Gaussian equilibrium function [[7](#ref7)]

$$\|p(\cdot,t)-p_{eq}\| \leq C e^{\normalsize -t}, \tag{4}$$

where $C$ is some constant with respect to time and $f_{eq}$ is a Normal (or Gaussian) distribution

$$
p_{eq}(x) = \frac{1}{(2\pi)^{d/2}} e^{\normalsize -\|x\|^2/2},
$$

see Figure 2. The idea behind generative models is to use the Fokker-Planck equation to transform the initial data onto a Gaussian and somehow learn to reverse this process. We will focus on the reverse process in the next section.

## Learning to reverse the Fokker-Planck equation 

While the particles perspective is valuable for numerical purposes, many works which study diffusion models adopts it quickly, sometimes neglecting the original Fokker-Planck equation. As we will see studying directly the Fokker-Planck equation allows to understand easily certain properties of diffusion models. The derivation of the reverse process is for example very natural in this case.

<div style="display: flex; flex-direction: row;">
    <img src="/assets/images/diffusion/moons_euler.gif" style="width: 50%;" />
    <img src="/assets/images/diffusion/swissroll_euler.gif" style="width: 50%;" />
</div>

**Figure 3:**  *Deterministic sampling ($\lambda=0$) using equation (8)<!--ref-->.*

 To reverse the Fokker-Planck equation let's write the backward equation of (3)<!--ref--> which simply consists in doing  the change of variable $t \rightarrow -t$ and hence putting a minus sign in front of the right hand side

$$ \partial_t p(x,t) = -\nabla \cdot (x p(x,t))- \Delta p (x,t). \tag{5}$$

For notational convenience we still denote the time $t$ here (and not $-t$) and we consider the equation (5)<!--ref--> running from $t=0$ to $t=T$ (and not from $t=T$ to $t=0$). Unfortunately it is not possible to reverse directly the equation (3)<!--ref-->. The equation is indeed not well-posed due to the minus sign in front of $\Delta p$. Intuitively since any initial solution converges toward the equilibrium, once we get to the equilibrium it is not possible to know where we started from. This is where the application of machine learning becomes essential. The goal here is to learn some special quantity during the forward process to reverse the equation. To understand how to do it we only need to use simple operations on differential operators.

First we notice that since $\nabla \log p = \nabla p / p$ and $\nabla \cdot \nabla p= \Delta p$ 

$$ \Delta p = \nabla \cdot (p \nabla \log p).  \tag{6}$$

Therefore equation (5)<!--ref--> can be written under the form 

$$
 \partial_t p(x,t) = -\nabla \cdot \Big(x p(x,t) + (1+\lambda)p(x,t)\nabla \log p(x,t) \Big) + \lambda \Delta p(x, t). \tag{7}
$$

where $\lambda\geq 0$ is a positive constant. Assuming that $\nabla \log p$ is known it is now possible to solve this equation (note that compare to equation (5)<!--ref--> there is now a positive sign in front of $\Delta p$). Hence the role of the machine learning framework will be to learn $\nabla \log p$ with a neural network depending on some parameters $\theta$ which we denote as

$$s_\theta(x,t) \approx \nabla \log p(x,t).$$

The neural network $s_{\theta}$ is commonly referred to as the score function. For image data it is frequently chosen to be a U-net [[10](#ref10)] but we won't go into the specifics details of this architecture. Once $s_{\theta}$ is trained we can use equation (2)<!--ref--> with $g(x,t) = -x-(1+\lambda)s_{\theta}(x,t)$ and $h(x)= \lambda$ to samples the particles from $p_{eq}$ (i.e. denoise the data)

$$
dx = (x + (1+\lambda) s_{\theta}(x,t))dt + \sqrt{2 \lambda dt} z, \quad z \sim \mathcal{N}(0, I).  \tag{8} 
$$


Note that when $\lambda =0$ the equation (8)<!--ref--> is a simple ordinary differential equation with no stochastic term while for $\lambda>0$ a stochastic term appear. In the literature [[1](#ref1), [2](#ref2), [6](#ref6)] the choices are often either $\lambda=0$ for the deterministic sampling or $\lambda=1/2$ based on a work from Anderson [[8](#ref8)] but it is actually possible to consider any $\lambda \geq 0$. 

![Generation process with Fokker-Planck](/assets/images/diffusion/gen_fp.png)

**Figure 4:** *Learning process of diffusion models. The Fokker-Planck is reversed thanks to $\nabla \log p$. In this example the deterministic form of the reversed equation is used.*



## Learning the score function $s_{\theta}$

So, in essence, diffusion models focus on learning the score function $s_\theta(x,t) \approx \nabla \log p(x,t)$ of a Fokker-Planck equation. However for practical machine learning applications computing directly $\nabla \log p$ is intractable. The trick [[9](#ref9)] is to use the conditional log probability $\nabla \log p( \cdot \vert x_0)$ where $x_0$ are data points of the initial probability distribution $p_0$. 

**Proposition 1:** The argmin of

$$
\argmin_\theta \int \|\nabla \log p (x, t) - s_{\theta}(x,t)\|^{2} p(x,t) dxdt, \tag{9}
$$

is the same as the argmin of

$$
\argmin_{\theta} E_{x_{0}\sim p_{0}}\Big(\int \|\nabla \log p (x,t \vert  x_{0}) - s_{\theta}(x,t)\|^{2}p (x,t \vert  x_{0})dx dt\Big) \tag{10}
$$

<details>
<summary>Click to display the proof</summary>

Expanding the square norm in (9)<!--ref--> and dropping the term which does not depend on $\theta$ one gets

$$
         \int \big(-2\nabla \log p(t, x) \cdot s_{\theta}(t, x) +\|s_{\theta}(t, x)\|^2 \big) p(t, x) d x.
$$

In the same way expanding the term (10)<!--ref--> in the same way and dropping the term which does not depend on $\theta$ one gets

$$
         \mathbb{E}_{x_{0} \sim p_0}
        \Big(\int_{0}^T \int_{\mathbb{R}^d}
        \big(-2\nabla_{x} \log p(t, x \vert  x_{0}) \cdot s_{\theta}(t, x) + \| s_{\theta}(t, x) \|^2\big) p(t, x \vert  x_{0})  d x d t\Big).
$$

Then we use the equalities $p\nabla \log p = \nabla p$ and $\mathbb{E}_{x_{0} \sim p_{0}}\big(p(t, x \vert  x_{0})\big) = p(t, x)$ to obtain

$$
         \int \big(-2\nabla \log p(t, x) \cdot s_{\theta}(t, x) +\|s_{\theta}(t, x)\|^2 \big) p(t, x) d x.
$$

We recover the expression which was obtained with the original formulation and conclude that the two argmin coincide. &#x25A0;

</details>

The previous proposition is central regarding the training of diffusion models because we can actually derive an analytical solution for $p(\cdot \vert x_0)$ and hence $\nabla \log p(\cdot \vert x_0)$. 

**Proposition 2:** If $p$ satisfies the Fokker-Planck equation (5)<!--ref--> then

$$
p(x,t \vert  x_0) = \dfrac{1}{(2  \pi \sigma_t^2)^{d/2} }\exp\big[(x-\mu_{t, x_0})^2/(2 \sigma_t^2)\big], \tag{11}
$$

where $\mu_{t, x_0} = x_{0} e^{-t}$, $\sigma_t = \sqrt{1-e^{-2t}}$.

<details>
<summary>Click to display the proof</summary>

It is possible to derive the solution (11)<!--ref--> from scratch using Fourier transform. A more direct way is to inject the solution (11)<!--ref--> into the Fokker-Planck equation (5)<!--ref--> and check that the equality is satisfied. &#x25A0;

</details>

We can then directly apply $\nabla \log$ to (11)<!--ref--> and obtain:

$$
\nabla \log p (x, t \vert  x_0) = - \frac{x - \mu_{t, x_0}}{\sigma_t^2}. \tag{12} 
$$

By combining Proposition 1 and equation (11)<!--ref-->-(12)<!--ref--> it is therefore possible to successfully learn the score function $s_{\theta}$.

In practice the loss is written as 

$$
\mathcal{L} := \min_{\theta} \mathbb{E}_{x_{0} \sim p_{0}} \Big(\int \frac{\alpha(t)}{\sigma_{t}^2} \|\frac{\mu_{t, x_{0}} - x}{\sigma_{t}} - \sigma_{t} s_{\theta}(x,t) \|^{2}  p(x,t \vert  x_{0})dt dx \Big), \tag{13}
$$

where $p(\cdot \vert  x_{0})$ is the probability distribution (11)<!--ref-->, $\alpha(t)$ is a time dependent function and often for diffusion models one takes 

$$\alpha(t) = \sigma_{t}^2, \tag{14}$$

which is a natural choice given it allows to remove the singularity in the loss (13)<!--ref-->. By using the form of $p(\cdot \vert x_{0})$ given in (11)<!--ref--> and doing the change of variable $z=(x-\mu_{t, x_{0}})/\sigma_{t}$ in (13)<!--ref--> one finally gets

$$
\mathcal{L} := \min_{\theta} \mathbb{E}_{x_{0} \sim p_{0}, \ z \sim \mathcal{N}(0,I)} \Big(\int \|z + \sigma_{t} s_{\theta}(\mu_{t, x_{0}} + \sigma_{t} z, t) \|^{2}dt \Big),
$$

which is the loss used during the training of diffusion models.

## Design choices for diffusion models

There are several design choices for diffusion models which are crucial in order to make them work. In this section, we will examine the key aspects and provide some insights into their justifications.

### The manifold hypothesis and the score $s_{\theta}$

 First let's look at the function $s_{\theta}$. This function is supposed to learn $\nabla \log p(x,t)$ for all $t \geq 0$. While there is no issue for $t>0$ there might be a problem for $t=0$ due to the Manifold Hypothesis. The Manifold Hypothesis is a guiding principle in machine learning and high-dimensional data analysis that suggests real-world high-dimensional data (like images, text, and sound) lie on low-dimensional manifolds embedded within the high-dimensional space. For diffusion models this means that the gradient of the initial density $p_{0}$ (and therefore also $\nabla \log p_{0}$) might not be well-defined (or equal to $+\infty$). For this reason $s_{\theta}$ is designed as follow 

$$
s_{\theta} = \frac{u_{\theta}}{\sigma_{t}}.
$$

This choice might seem arbitrary at first but notice that since $\sigma_{t} \rightarrow 0$ when $t \rightarrow 0$ this makes the score $s_{\theta}$ singular in the limit $t \rightarrow 0$ which is the desired behavior when the probability $p_{0}$ lies on a manifold

$$ 
s_{\theta}(x, t) \xrightarrow[t \to 0]{} + \infty
$$


 Note that this singularity might require extra care when integrating the backward Fokker-Planck equation.  We will see this in the next subsection.

### Time scheduling

One crucial aspect of diffusion models is time scheduling. When considering the Fokker-Planck equation  this is equivalent to the following change of variable [[1](#ref1)]

$$
t_{\text{new}} = \frac{1}{20 T}t + \frac{19.9}{4 T^{2}} t^{2}. \tag{15}
$$            


<div style="display: flex; flex-direction: row;">
    <img src="/assets/images/diffusion/t2.png" style="width: 50%;" />
    <img src="/assets/images/diffusion/exp_t2.png" style="width: 50%;" />
</div>

**Figure 5:** *Change of variable in time. On the left $t$ vs $t_{new}$. On the right $e^{-t}$ vs $e^{-t_{new}}$.*

It might not be clear at first why this change of variable is needed and many presentations remain elusive about it. Here are some of the reasons which might justify the introduction of the new time dependence:

**- Balance for the temporal weight $\alpha(t) = \sigma_{t}^{2}$ in the loss function:** In (13)<!--ref--> the loss function $\mathcal{L}$ is modified by selecting $\alpha(t) = \sigma_{t}^2$ as the weighting function, altering its time dependence. Given that $\sigma(0) = 0$ and $\sigma_{t} \rightarrow 1$ when $t \rightarrow +\infty$, this results in a diminished emphasis on earlier times compared to the baseline choice of $\alpha = 1$. The variable transformation introduced in (15)<!--ref--> might compensates by reallocating some of the weight back to the initial times see Figure 5.

**- Dealing with the exponential convergence toward equilibrium:** As shown in equation (4)<!--ref--> the function $p(\cdot,t)$ converges exponentially fast toward the equilibrium 

$$\|p(\cdot,t)-p_{eq}\| \leq C e^{-t}.$$

Numerically it might be difficult to learn or sample a density function $p$ that varies exponentially fast. Time scheduling provides a work around that as it will slow down the exponential convergence. As shown on the right of Figure 5, the decay of $e^{-t_{\text{new}}}$ is slower than $e^{-t}$ rendering it more favorable for numerical purposes.
 
 **- Dealing with the singularity of the score function:** as we have seen in the previous section the score function $s_{\theta}$ becomes singular when $t \rightarrow 0$. During inference it might therefore be wise to take extra care for integration near for small times. The time scheduling also allows to do that by putting more weight near initial times as shown in Figure 5.

## Training and sampling algorithms

We are now ready to give the training and sampling algorithms for diffusion models. As depicted in Figure 6, the training and sampling procedures encompassed all the concepts introduced so far.

<img src="/assets/images/diffusion/algo.png">

**Figure 6:** *Training and sampling algorithms for diffusion models.*

And that's it we can now train and sample from diffusion models. Below are some examples of the sampling procedure for some 2d toy distributions both in the deterministic and stochastic cases.

<div style="display: flex; flex-direction: row;">
    <img src="/assets/images/diffusion/moons_euler.gif" style="width: 50%;" />
    <img src="/assets/images/diffusion/swissroll_euler.gif" style="width: 50%;" />
</div>
<div style="display: flex; flex-direction: row;">
    <img src="/assets/images/diffusion/moons_anderson.gif" style="width: 50%;" />
    <img src="/assets/images/diffusion/swissroll_anderson.gif" style="width: 50%;" />
</div>

**Figure 7:** *Sampling the density. Top: deterministic sampling ($\lambda =0$), bottom: stochastic sampling ($\lambda=1/2$).*

A more realistic example is provided in Figure 8 where the dataset is made of facial images. The animations represent the sampling process where each image is interpreted as a high-dimensional point sampled from a Gaussian distribution and mapped back to the original distribution of images through the application of a diffusion model.


<div style="display: flex; flex-direction: row;">
    <img src="/assets/images/diffusion/face_0.gif" style="width: 33%;" />
    <img src="/assets/images/diffusion/face_1.gif" style="width: 33%;" />
    <img src="/assets/images/diffusion/face_2.gif" style="width: 33%;" />
</div>

**Figure 8:** *Images generation using a diffusion model trained on facial images.*

## Extension to other types of equilibria


We have seen that the fundamental property behind diffusion models is the convergence toward an equilibrium. As there exists many other physical processes with this property, a natural question to ask is whether we could use some of them for generative modeling purpose. We will review some approaches which rely on equilibria in the following subsections.

### The non-homogeneous case and some extensions  

In physics, the Fokker-Planck equation, as introduced in equation (3)<!--ref-->, is interpreted over the velocity space for a system characterized by a density distribution homogeneous in space. For the sake of simplicity, our earlier discussions employed the notation $x$ to refer to the unknown velocity variable. As we transition to examining the non-homogeneous case we will now use the notation $x$ for space and $v$ for velocities. The non-homogeneous Fokker-Planck equation for a density function $p(x,v,t)$ can be written under the form

$$\partial_t p + v \nabla_{\mathbf{x}} p - x \nabla_{v} p = \nabla_{v} \cdot (vp + \nabla_{v} p), \tag{16}$$

where $x, v \in \mathbb{R}^d$ are the space and velocity variables respectively. Using this equation for generative modeling is the approach followed in [[13](#ref13)] (they also introduce some additional parameters which does not change the general behavior of the probability density function compare to (16)<!--ref-->). The derivation of the generative method in the non-homogeneous case is close to the presentation given above for diffusion models see [[13](#ref13)] for details. 

For practical applications, one of the main distinction with classical diffusion models is that the data are now represented in the space variables while the network is learning $\nabla_{v} \log p$. The gradient is therefore taken with respect to the velocity and not the space (i.e. data) variables. It can help as in practice the distribution function may be smoother with respect to the $v$ than $x$ (and in fact the spatial gradient may be ill-defined for $t=0$).

Back to our equilibrium perspective, let's now look at some research papers which study the convergence of the non-homogeneous Fokker-Planck equation (16)<!--ref-->. In fact this equation also converges toward a Gaussian equilibrium both in space and velocity

$$
p_{eq}(x, v) = \frac{1}{(2\pi)^{d}} e^{\normalsize -(\|x\|^2+\|v\|^2)/2},
$$

The convergence result can be stated as follow: for any $\epsilon>0$ there exists $c_{\epsilon}$ such that

$$
        \| p - p_{eq}\|^{2} \leq c_{\epsilon}  e^{\normalsize -(\frac{2}{\gamma}-\epsilon)t}
$$

A proof can be found in [[11](#ref11)] see also [[12](#ref12)] for a more recent result without any $\epsilon$. From a physical perspective it is therefore natural to use the non-homogeneous Fokker-Planck equation for generative modeling. One particularly interesting point when considering convergence results such as the ones presented in [[11](#ref11), [12](#ref12)] is that they not only concerns the equation (16)<!--ref--> but also more general systems of the form

$$
        \partial_{t} p(w, t) = \nabla \cdot \big(\mathbf{C} w p(w, t) + \mathbf{D} \nabla p(w, t)), \\
$$

where $\mathbf{C}$ and $\mathbf{D}$ are two matrices and $w$ is the (possibly augmented) unknown variable. Hence, what we have only seen so far correspond two particular choices for the matrices $\mathbf{C}$ and $\mathbf{D}$ but there are actually various other diffusion processes which could be considered for generative purposes.

### Gradient flows

Let $\mathcal{F}$ be some functional defined on the space of probability measures and consider the following equation

$$
\partial_{t} p = \nabla \cdot (p \nabla \frac{\delta \mathcal{F}}{\delta p}),  \tag{17}
$$

here the notation $\frac{\delta \mathcal{F}}{\delta p}$ is used to represent the first variation of $\mathcal{F}$. The equality (17)<!--ref--> is a general form of equation which may admit an equilibrium for some specific choice of $\mathcal{F}$. In practice the equilibrium will be the probability distribution which makes $\frac{\delta \mathcal{F}}{\delta p}$ constant. For instance, convergence toward equilibrium is proven [[14](#ref14)] for $\frac{\delta \mathcal{F}}{\delta p}$ under the general form 

$$
\frac{\delta \mathcal{F}}{\delta p}(p) = U'(p) + V + W * p,
$$

where $U$, $V$ and $W$ should satisfy some technical assumptions see [[14](#ref14)] for details.

As an example let 

$$
\frac{\delta \mathcal{F}}{\delta p} = \log p + \|x\|^{2}/2, \tag{18}
$$

 then by injecting the above equality in (17)<!--ref--> we recover the Fokker-Planck equation

$$\partial_t p(x,t) = \nabla \cdot (x p(x,t))+ \Delta p (x,t).$$

However the Fokker-Planck equation is a very particular example of the equation (17)<!--ref--> as it is linear. For other choices of $\mathcal{F}$ the equation (17)<!--ref--> will become non-linear making it much more difficult to solve in practice. Many research papers tackle this problem by using the so-called gradient flow approach [[15](#ref15), [16](#ref16), [17](#ref17), [18](#ref18), [19](#ref19), [20](#ref20)]. That is considering the minimization problem

$$
\min_{p \in \mathcal{P}} \mathcal{F}(p),
$$

and follow the steepest descent direction of the function $\mathcal{F}(p)$ with respect to the gradient flow Riemannian metric induced by some special distance called the 2-Wasserstein distance. We will not go into details as this approach requires knowledge of optimal transport [[21](#ref21)]. 

If the gradient flow approach is also able to reverse the Fokker-Planck equation with the particular choice (18)<!--ref-->, one might wonder why diffusion models seems more popular at the moment. One of the reason might be that, even if the gradient flow approach is much more general (it can handle non-linear equations), it does not exploit the linearity of the Fokker-Planck equation contrary to diffusion models. Numerically, this could lead to enhanced efficiency for diffusion models.

### Flow matching / stochastic interpolants

Flow matching / stochastic interpolants [[22](#ref22), [23](#ref23), [24](#ref24)] are generative methods which used explicit path between any pair of probability densities $p_{0}$ and $p_{1}$. The main idea is to define an interpolation $x_{t}$ between the particles of the two densities, as an example one might think at the simple linear case

$$
x_{t}( x_{0}, x_{1}) = (1-t)x_{0} + t x_{1}, \qquad x_{0} \sim p_{0}, \ x_{1} \sim p_{1}, \ t \in [0,1]. \tag{19}
$$

The idea is then to learn a neural network $v_{\theta}$ by minimizing the objective function

$$
\min_{\theta} \mathbb{E}_{t, \ x_{0} \sim p_{0}, \ x_{1} \sim p_{1}}\Big(|v_{\theta}(x_{t}(x_{0}, x_{1}), t)- \partial_{t} x_{t}(x_{0},x_{1}) |^{2}  \Big),
$$

which can then be used as a flow to transport particles between the two densities by following the ODE

$$
\frac{d}{dt}X_{t} = v_{\theta}(X_{t}, t).
$$

When an explicit path is given, the flow matching approach can therefore be used to learn a flow between the two densities. In particular, this is the case for diffusion models where the interpolating path can be written as

$$
x_{t}(x_{0}, x_{1}) =  e^{-t}x_{0} + \sqrt{1-e^{-2t}} z, \qquad z \sim p_{1}:=\mathcal{N}(0, I).
$$

Using the above interpolation for flow matching it is possible to recover the density of the diffusion approach. The flow matching framework however is more general in the sense that other interpolations such as the one presented in (19)<!--ref--> can be used connecting arbitrary densities (with the possibility that none of them is Gaussian) see Figure 9. 

<div style="text-align: center;">
<img src="/assets/images/diffusion/stochastic_interpolant.gif" alt="density interpolation" width="50%" />
</div>

**Figure 9:** *Interpolation between two densities using the flow matching / stochastic interpolant framework.*


It also enables the consideration of other physical systems for generative purpose. As an example consider the BGK model [[25](#ref25)] which is one of the simplest physical model to describe an exponential convergence of a probability distribution $p$ toward an equilibrium $p_{eq}$

$$
\partial_{t} p(x,t) = p_{eq}(x) - p(x,t).\tag{20}
$$

One can check that $e^{-t}p + (1-e^{-t})p_{eq}$ is a solution to (20)<!--ref--> and the interpolation

$$
x_{t}(x_{0}, x_{1}) =  e^{-t}x_{0} + (1-e^{-t}) z, \qquad z \sim p_{eq}:=\mathcal{N}(0, I),
$$

can therefore be used to learn a flow between the densities $p$ and $p_{eq}$ following the BGK density. Finally note that the non-homogenous case considered before can also be written under this formalism.

Even if the flow matching approach is more general than diffusion models not all equilibrium methods can be recast into this framework see Figure 10. For the gradient flows method described in the previous section for example, it is not possible, in general, to give an explicit interpolating path between the two densities.

<div style="text-align: center;">
<img src="/assets/images/diffusion/diagram.png" alt="diagram" width="80%" />
</div>

**Figure 10:** *Diagram of flow matching and equilibrium methods for generative modeling purpose.*


### Extension to discrete datasets

The approaches we have presented so far are designed to be used for continuous data. A natural question is how to generalize diffusion models to discrete datasets. If we look at diffusion models from the perspective of reversing a continuous process which converges toward an equilibrium, a natural extension would be then to consider discrete probability distributions which converge toward a discrete equilibrium. Such extensions have been studied for example in [[26](#ref26), [27](#ref27)]. Basically the discrete probability distribution satisfies the following equation

$$
\partial_{t} p = M_{t} p,
$$

where $p \in \mathbb{R}^d$ is a discrete probability distribution and $M_{t} \in \mathbb{R}^{d \times d}$ is a time dependent markov transition matrix. For example if $M$ is doubly stochastic with positive entries, then it can be shown that the corresponding probability distribution $p$ converges toward a uniform distribution. For instance in [[26](#ref26)] one possible choice is given by

$$
M_{t}^{ij} =
\begin{cases}
    &\beta_{t} / d, \quad &\text{if } i\ne j, \\
    &1-\sum_{l \ne i} M_{t}^{il}, \quad &\text{if } i=j,
\end{cases}
$$

where $\beta_{t} \in [0,1]$ is a time dependent scalar. This choice ensures the convergence of $p$ toward a uniform distribution. It then remains to learn how to reverse the process we refer to [[26](#ref26), [27](#ref27)] for details.

<div style="display: flex; flex-direction: row;">
    <img src="/assets/images/diffusion/discrete_forward.gif" style="width: 50%;" />
    <img src="/assets/images/diffusion/discrete_euler.gif" style="width: 50%;" />
</div>

**Figure 11:** *Discrete diffusion model applied on a discrete toy dataset. Left: convergence of the initial distribution toward the uniform equilibrium. Right: Sampling procedure.*

## Citation & Code

Cited as:


> Morel, Guillaume. (Mar. 2024). Diffusion models from an equilibrium perspective. https://morel-g.github.io/2024/03/25/diffusion-models-from-an-equilibrium-perspective.html


Or


> @misc{Morel24,
>  author = {Morel, Guillaume},
>  title = {Diffusion models from an equilibrium perspective},
>  year = {2024},
>  month = {Mar},
>  url = "https://morel-g.github.io/2024/03/25/diffusion-models-from-an-equilibrium-perspective.html"
> }

The code utilized for generating some examples showcased in this post, along with supplementary materials, is available [here](https://github.com/morel-g/generative-models). 

****

## References

<a id="ref1"></a> 1. Song, Yang; Ermon, Stefano. "Generative modeling by estimating gradients of the data distribution." *Advances in Neural Information Processing Systems* 32 (2019). [URL](https://arxiv.org/abs/1907.05600).

<a id="ref2"></a> 2. Ho, Jonathan; Jain, Ajay; Abbeel, Pieter. "Denoising Diffusion Probabilistic Models." In *Advances in Neural Information Processing Systems*, vol. 33, pp. 6840-6851, 2020. Curran Associates, Inc. [URL](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf).

<a id="ref3"></a> 3. Sohl-Dickstein, Jascha; Weiss, Eric; Maheswaranathan, Niru; Ganguli, Surya. "Deep Unsupervised Learning using Nonequilibrium Thermodynamics." In *Proceedings of the 32nd International Conference on Machine Learning*, pp. 2256-2265. Vol. 37. Proceedings of Machine Learning Research. [URL](https://proceedings.mlr.press/v37/sohl-dickstein15.html).

<a id="ref4"></a> 4. Song, Yang; Sohl-Dickstein, Jascha; Kingma, Diederik P; Kumar, Abhishek; Ermon, Stefano; Poole, Ben. "Score-Based Generative Modeling through Stochastic Differential Equations." In *International Conference on Learning Representations*, 2021. [URL](https://openreview.net/forum?id=PxTIG12RRHS).

<a id="ref5"></a> 5. Weng, Lilian. (Jul 2021). What are diffusion models? Lil’Log. *Blog post*. [URL](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).

<a id="ref6"></a> 6. Song, Yang. Generative Modeling by Estimating Gradients of the Data Distribution. *Blog post*. [URL](https://yang-song.net/blog/2021/score/)

<a id="ref7"></a> 7. Toscani, Giuseppe. "Entropy production and the rate of convergence to equilibrium for the Fokker-Planck equation." Quarterly of Applied Mathematics 57 (1999): 521-541. [URL](https://www.ams.org/journals/qam/1999-57-03/S0033-569X-1999-1704435-X/S0033-569X-1999-1704435-X.pdf)

<a id="ref8"></a> 8. Anderson, Brian D.O. "Reverse-time diffusion equation models." *Stochastic Processes and their Applications* 12.3 (1982): 313-326. ISSN: 0304-4149. [URL](https://www.sciencedirect.com/science/article/pii/0304414982900515).

<a id="ref9"></a> 9. Vincent, Pascal. "A connection between score matching and denoising autoencoders." *Neural Computation* 23.7 (2011): 1661-1674. MIT Press. [URL](https://www.iro.umontreal.ca/~vincentp/Publications/DenoisingScoreMatching_NeuralComp2011.pdf).

<a id="ref10"></a> 10. Ronneberger, Olaf, Fischer, Philipp, and Brox, Thomas. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *CoRR* abs/1505.04597 (2015). [URL](http://arxiv.org/abs/1505.04597).

<a id="ref11"></a> 11. Arnold, Anton and Erb, Jan. "Sharp entropy decay for hypocoercive and non-symmetric Fokker-Planck equations with linear drift." *arXiv* (2014). [URL](https://arxiv.org/abs/1409.5425).

<a id="ref12"></a> 12. Arnold, Anton, Einav, Amit, and Wöhrer, Tobias. "On the rates of decay to equilibrium in degenerate and defective Fokker–Planck equations." *Journal of Differential Equations* 264.11 (2018): 6843-6872. ISSN: 0022-0396. [URL](https://www.sciencedirect.com/science/article/pii/S0022039618300767).

<a id="ref13"></a> 13. Dockhorn, Tim, Vahdat, Arash, and Kreis, Karsten. "Score-Based Generative Modeling with Critically-Damped Langevin Diffusion." In *International Conference on Learning Representations* (2022). [URL](https://arxiv.org/abs/2112.07068).

<a id="ref14"></a> 14. Carrillo, José A., McCann, Robert J., and Villani, Cédric. "Kinetic equilibration rates for granular media and related equations: entropy dissipation and mass transportation estimates." *Revista Matematica Iberoamericana* 19.3 (2003): 971-1018. [URL](https://projecteuclid.org/journals/revista-matematica-iberoamericana/volume-19/issue-3/Kinetic-equilibration-rates-for-granular-media-and-related-equations/rmi/1077293812.full).


<a id="ref15"></a> 15. Bunne, Charlotte, Meng-Papaxanthos, Laetitia, Krause, Andreas, and Cuturi, Marco. "Proximal Optimal Transport Modeling of Population Dynamics." *arXiv* (2022). [URL](https://arxiv.org/abs/2106.06345).


<a id="ref16"></a> 16. Mokrov, Petr, Korotin, Alexander, Li, Lingxiao, Genevay, Aude, Solomon, Justin, and Burnaev, Evgeny. "Large-Scale Wasserstein Gradient Flows." *arXiv* (2021). [URL](https://arxiv.org/abs/2106.00736).

<a id="ref17"></a> 17. Alvarez-Melis, David, Schiff, Yair, and Mroueh, Youssef. "Optimizing Functionals on the Space of Probabilities with Input Convex Neural Networks." *arXiv* (2021). [URL](https://arxiv.org/abs/2106.00774).

<a id="ref18"></a> 18. Yang, Zhuoran, Zhang, Yufeng, Chen, Yongxin, and Wang, Zhaoran. "Variational Transport: A Convergent Particle-Based Algorithm for Distributional Optimization." *arXiv* (2020). [URL](https://arxiv.org/abs/2012.11554).

<a id="ref19"></a> 19. Liutkus, Antoine, Şimşekli, Umut, Majewski, Szymon, Durmus, Alain, and Stöter, Fabian-Robert. "Sliced-Wasserstein Flows: Nonparametric Generative Modeling via Optimal Transport and Diffusions." *arXiv* (2019). [URL](https://arxiv.org/abs/1806.08141).

<a id="ref20"></a> 20. Bonet, Clément, Courty, Nicolas, Septier, François, and Drumetz, Lucas. "Efficient Gradient Flows in Sliced-Wasserstein Space." *arXiv* (2022). [URL](https://arxiv.org/abs/2110.10972).

<a id="ref21"></a> 21. Ambrosio, Luigi, Gigli, Nicola, and Savaré, Giuseppe. *Gradient Flows in Metric Spaces and in the Space of Probability Measures*. 2nd ed., Birkhäuser, 2008. Lectures in Mathematics ETH Zürich. ISBN: 978-3-7643-8722-8 978-3-7643-8721-1.

<a id="ref22"></a> 22. Lipman, Yaron, Chen, Ricky T. Q., Ben-Hamu, Heli, Nickel, Maximilian, and Le, Matt. "Flow Matching for Generative Modeling." arXiv (2023). [URL](https://arxiv.org/abs/2210.02747).

<a id="ref23"></a> 23. Albergo, Michael S., Boffi, Nicholas M., and Vanden-Eijnden, Eric. "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions." arXiv (2023). [URL](https://arxiv.org/abs/2303.08797)

<a id="ref24"></a> 24. Liu, Xingchao, Gong, Chengyue, and Liu, Qiang. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." arXiv (2022). [URL](https://arxiv.org/abs/2209.03003).

<a id="ref25"></a> 25. Bhatnagar, P. L., Gross, E. P., and Krook, M. "A Model for Collision Processes in Gases. I. Small Amplitude Processes in Charged and Neutral One-Component Systems." Phys. Rev. 94, no. 3 (May 1954): 511–525. doi:10.1103/PhysRev.94.511. [URL](https://link.aps.org/doi/10.1103/PhysRev.94.511)

<a id="ref26"></a> 26. Austin, Jacob, Johnson, Daniel D., Ho, Jonathan, Tarlow, Daniel, and van den Berg, Rianne. "Structured Denoising Diffusion Models in Discrete State-Spaces." *CoRR* abs/2107.03006 (2021). [URL](https://arxiv.org/abs/2107.03006).

<a id="ref27"></a> 27. Santos, Javier E., Fox, Zachary R., Lubbers, Nicholas, and Lin, Yen Ting. "Blackout Diffusion: Generative Diffusion Models in Discrete-State Spaces." *arXiv* (2023). [arXiv:2305.11089](https://arxiv.org/abs/2305.11089).
