.. raw:: html

   <script type="text/x-mathjax-config">
     MathJax.Hub.Config({
       extensions: ["tex2jax.js"],
       jax: ["input/TeX", "output/HTML-CSS"],
       tex2jax: {
         inlineMath: [ ['$','$'], ["\\(","\\)"] ],
         displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
         processEscapes: true
       },
       "HTML-CSS": { availableFonts: ["TeX"] }
     });
   </script>
   <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"">
   </script>

.. _paper: https://arxiv.org/abs/1812.04652

Algorithm Descriptions
======================

For all the algorithm descriptions, let $I(\\mathbf x)$ be the MR brain image under consideration where
$\\mathbf x \\in \[0,N\]\\times\[0,M\]\\times\[0,L\]\\subset \\mathbb{N}^3$ for $N,M,L \\in \\mathbb{N}$, the dimensions of $I$,
and let $B \\subset I$ be the corresponding brain mask (i.e., the set of indices
corresponding to the location of the brain in $I$).

If any of the descriptions here are unclear, please see the corresponding paper_ for a more concise, refined description.

Z-score
~~~~~~~

Z-score normalization uses the brain mask $B$ for the image $I$ to
determine the mean and standard deviation of the intensities inside the brain
mask, that is:
$$ \\mu = \\frac{1}{\|B\|} \\sum_{\\mathbf b \\in B} I(\\mathbf b) \\quad \\text{and} \\quad
\\sigma = \\sqrt{\\frac{\\sum_{\\mathbf b \\in B} (I(\\mathbf b) - \\mu)^2}{\|B\|-1}} $$
Then the Z-score normalized image is
$$ I_{\\text{z-score}}(\\mathbf x) = \\frac{I(\\mathbf x) - \\mu}{\\sigma}. $$

Fuzzy C-Means
~~~~~~~~~~~~~

Fuzzy C-means normalization uses a segmentation of a specified tissue (i.e., CSF, GM, or WM) to
normalize the entire image to the mean of the tissue. The procedure is as follows.
Let $T\\subset B$ be the tissue mask for the image $I$, i.e., $T$ is the set of indices
corresponding to the location of the tissue in the image $I$. Then the tissue mean is
$$ \\mu = \\frac{1}{\|T\|} \\sum_{\\mathbf t \\in T} I(\\mathbf t) $$
and the segmentation-based normalized image is
$$ I_{\\text{seg}}(\\mathbf x) = \\frac{c\\cdot I(\\mathbf x)}{\\mu} $$
where $c \\in \\mathbb{R}_{>0}$ is some constant. In this function, we use
three-class fuzzy c-means to get a segmentation of the tissue over the brain mask
$B$ for the T1-w image and we arbitrarily set $c = 1$.

Kernel Density Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~

KDE-based normalization estimates the empirical probability density function (pdf) of the
intensities of $I$ over the brain mask $B$ using the method of kernel density
estimation.

The KDE of the pdf for the intensity of the image is calculated as follows:
$$ \\hat{p}(x) = \\frac{1}{N\\cdot M \\cdot L \\cdot \\delta} \\sum_{i = 1}^{N\\cdot M \\cdot L} K\\left(\\frac{x - x_i}{\\delta}\\right)$$
where $x$ is an intensity value, $K$ is the kernel (a kernel is
essentially a non-negative function that integrates to 1), $\\delta$ is the
bandwidth parameter (that is, a smoothing parameter) which scales the kernel
$K$.

In our experiment, we use a Gaussian kernel and set $\\delta = 80$, which
was found to empirically determine a reasonable density estimate across various
datasets.

The kernel density estimate provides a smooth version of the histogram
which allows us to more robustly pick a the **mode** associated with the WM via a
combinatorial optimization routine. The found WM peak $p$ is then used to
normalize the entire image, in much the same way as the segmentation-based
normalization. That is,
$$ I_{\\text{kde}}(\\mathbf x) = \\frac{c \\cdot I(\\mathbf x)}{p} $$
where $c \\in \\mathbb{R}_{>0}$ is some constant. In this package, we
arbitrarily set $c = 1$.

Piecewise Linear Histogram Matching (Nyul & Udupa)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Piecewise (affine) histogram-based normalization (which we will denote as HM for
brevity)—proposed by Nyul and Udupa [1]_—addresses the normalization problem by learning a
standard histogram for a set of contrast images and mapping the intensities of each
image to this standard histogram. The standard histogram is learned through the
demarcation of pre-defined landmarks of interest.

For instance, Shah et al. [2]_ defined landmarks as intensity percentiles at
$1,10,20,\\ldots,90,99$ percent (where the intensity values below 1% and above 99% are
discarded as outliers). We use these landmarks as the default in our implementation. The
standard scale must be have a pre-defined range, i.e., $[m_{\\text{min}}^s, m_{\\text{max}}^s]$.
In our experiment, we arbitrarily set $m_{\\text{min}}^s = 0$ and $m_{\\text{max}}^s = 100$.

Learning the standard histogram
"""""""""""""""""""""""""""""""

Let $\\mathbf I = \\{I_1,I_2,\\ldots,I_K\\}$ be a set of $K$ MR brain images of one contrast.
We calculate the following set of quantities $m_1^i$ and $m_{99}^i$, which are the 1% and 99%
intensity values for the image $I_i \\in \\mathbf I$. We then map all the intensity values of
$I_i$ with the following linear map $$ \\tilde I_i(\\mathbf x) = \\left(I_i(\\mathbf x) - m_1^i + m_{\\text{min}}^s\\right) \\left(\\frac{m_{\\text{max}}^s}{m_{99}^i}\\right) $$
which takes the intensities of $I_i$ to the range $[m_{\\text{min}}^s, m_{\\text{max}}^s]$ excluding outliers.
Then we calculate the deciles for the new image $\\tilde I_i$, i.e., the set
$\\{\\tilde m_{10}^i,\\tilde m_{20}^i,\\ldots,\\tilde m_{90}^i\\}$ (note that $\\tilde
m_0^i = m_{\\text{min}}^s$ and $\\tilde m_{100}^i = m_{\\text{max}}^s$). This is
done over every image $I_i \\in \\mathbf I$ and the mean of each corresponding
value is the learned landmark for the standard histogram. That is, for $n \\in
\\{10,20,\\ldots,90\\}$, we have
$$ m_n^s = \\frac{1}{K} \\sum_{i=1}^{K} \\tilde m_n^i $$
and the standard scale landarks is the set
$\\{m_{\\text{min}}^s,m_{10}^s,\\ldots,m_{90}^s,m_{\\text{max}}^s\\}$.

Normalizing new images
""""""""""""""""""""""

For a test image $I$, the transform for the normalization is done by first calculating
the set of percentiles $\\{m_{1},m_{10},m_{20},\\ldots,m_{90},m_{99}\\}$. These
values are then used to segment the image into deciles, i.e., we define 10 non-overlapping
sets of indices $D_{i,j} = \\{\\mathbf x \\mid  m_i \\le I(\\mathbf x) < m_j\\}$ where
$i,j \\in \\{1,10,20,\\ldots,90,99\\}$ and restricting $j$ to equal the next value
in the set greater than $i$. We then piecewise linearly map the
intensities associated with these deciles to the corresponding decile on the
standard scale landmarks. Noting that each $D_{i,j}$ is disjoint from the other,
the normalized image is then defined as
$$ I_{\\text{nu}} = \\bigcup_{\\substack{i,j \\in \\{1,10,20,\\ldots,90,99\\}\\newline i\\neq j, i \\le j + 10}} \\left(\\frac{I(D_{i,j}) - m_i}{m_j - m_i}\\right) \\left(m_j^s - m_i^s\\right) + m_i^s. $$

WhiteStripe
~~~~~~~~~~~

WhiteStripe intensity normalization [3]_ attempts to do a Z-score normalization based on the
intensity values of normal appearing white matter (NAWM). The NAWM is found by smoothing the
histogram of the image (i.e., KDE) and selecting the mode of the distribution (for T1-w images).
Let $p$ be the intensity associated with the mode. The "white stripe" is then defined as the 10%
segment of intensity values around $\\mu$. That is, let $F(x)$ be the cdf of the
specific MR image $I(\\mathbf x)$ inside its brain mask $B$, and define $\\tau =
5\\%$. Then, the white stripe $\\Omega_\\tau$ is defined as the set
$$ \\Omega_\\tau = \\left\\{I(\\mathbf x) \\mid F^{-1}\\left(F(\\mu) - \\tau\\right) < I(\\mathbf x) < F^{-1}\\left(F(\\mu) + \\tau\\right)\\right\\}. $$
Let $\\sigma$ be the sample standard deviation associated with $\\Omega_\\tau$.
Then the WhiteStripe normalized image is
$$ I_{\\text{ws}}(\\mathbf x) = \\frac{I(\\mathbf x) - \\mu}{\\sigma}. $$

RAVEL
~~~~~

RAVEL normalization [4]_ attempts to improve upon the result of WhiteStripe by
removing unwanted technical variation, e.g., scanner effects. RAVEL assumes the
set of images can be expressed in the additive model
$$ V = \\alpha 1^T + \\beta X^T + \\gamma Z^T + R $$
where $V$ is a population of WhiteStripe normalized images of the same contrast,
$\\alpha 1^T$ is the average scan, $\\beta X^T$ represents known clinical
covariates (e.g., age, gender), $\\gamma Z^T$ represents the unknown, unwanted
factors (i.e., the technical variability), and $R$ is the matrix of residuals.

Since this model is assumed, if we can determine voxels in the MR image where
there are no clinical covariates, then we can solve for the unwanted factors
$\\beta X^T$ through simple linear regression. The authors, Fortin et al., assume
that CSF is not associated with these clinical covariates and uses the voxels
associated with CSF as the control voxels. Then if the average scan is removed,
the voxels associated with the CSF is of the form
$$ V_c = \\gamma Z^T + R $$
where $V_c$ are the set of control (CSF) voxels.

Note that we can rewrite $V_c$ as
$$ V_c = U \\Sigma W^T $$
through the SVD. If $W$ is an $n\\times n$ matrix of right singular vectors.
Then we can use $b<n$ right singular vectors to form an orthogonal basis for the
unwanted factors $Z$ [5]_. That is, we use $W_b$ as the estimate of
$Z$, where $W_b$ are the select $b$ right singular vectors. We then do
voxel-wise linear regression to estimate the coefficients $\\gamma$. Then the
RAVEL normalized image is simply
$$ I_{\\text{ravel}}(\\mathbf x) = I_{\\text{ws}}(\\mathbf x) - \\gamma_{\\mathbf x} Z^T. $$
where $\\gamma_{\\mathbf x}$ are the coefficients of unwanted variation associated
with the voxel $\\mathbf x$ found via linear regression. In our experiments, we follow the original
paper [4]_ and set $b=1$ to be the first singular
vector (the first right singular vector is highly correlated (>95%)
with the mean intensity of the CSF).

References
~~~~~~~~~~

.. [1] L. G. Nyúl, J. K. Udupa, and X. Zhang, “New Variants of a Method of MRI Scale Standardization,” IEEE Trans. Med. Imaging, vol. 19, no. 2, pp. 143–150, 2000.

.. [2] M. Shah, Y. Xiao, N. Subbanna, S. Francis, D. L. Arnold, D. L. Collins, and T. Arbel, “Evaluating intensity normalization on MRIs of human brain with multiple sclerosis,” Med. Image Anal., vol. 15, no. 2, pp. 267–282, 2011.

.. [3] R. T. Shinohara, E. M. Sweeney, J. Goldsmith, N. Shiee, F. J. Mateen, P. A. Calabresi, S. Jarso, D. L. Pham, D. S. Reich, and C. M. Crainiceanu, “Statistical normalization techniques for magnetic resonance imaging,” NeuroImage Clin., vol. 6, pp. 9–19, 2014.

.. [4] J. P. Fortin, E. M. Sweeney, J. Muschelli, C. M. Crainiceanu, and R. T. Shinohara, “Removing inter-subject technical variability in magnetic resonance imaging studies,” Neuroimage, vol. 132, pp. 198–212, 2016.

.. [5] J. T. Leek and J. D. Storey, “Capturing heterogeneity in gene expression studies by surrogate variable analysis,” PLoS Genet., vol. 3, no. 9, pp. 1724–1735, 2007.
