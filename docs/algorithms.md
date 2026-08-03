# How the methods work

For all descriptions, let $I(\mathbf{x})$ be the MR brain image under consideration where $\mathbf{x} \in [0,N] \times
[0,M] \times [0,L] \subset \mathbb{N}^3$, and let $B \subset I$ be the brain mask (the set of voxels in the brain).

## Z-score

Z-score normalization uses the brain mask $B$ to determine the mean and standard deviation of the intensities inside the
brain:

$$ \mu = \frac{1}{|B|} \sum_{\mathbf{b} \in B} I(\mathbf{b}) \quad \text{and} \quad \sigma =
\sqrt{\frac{\sum_{\mathbf{b} \in B} (I(\mathbf{b}) - \mu)^2}{|B| - 1}} $$

Then the z-score normalized image is

$$ I_{\text{z-score}}(\mathbf{x}) = \frac{I(\mathbf{x}) - \mu}{\sigma}. $$

Z-score is not brain-specific and works for any anatomy.

## Fuzzy C-means (FCM)

FCM normalization scales the image so the mean of a chosen tissue class equals a constant. Let $T \subset B$ be the
tissue of interest (CSF, GM, or WM). The tissue mean is

$$ \mu = \frac{1}{|T|} \sum_{\mathbf{t} \in T} I(\mathbf{t}) $$

and the normalized image is

$$ I_{\text{fcm}}(\mathbf{x}) = \frac{c \cdot I(\mathbf{x})}{\mu} $$

where $c \in \mathbb{R}_{>0}$ is a constant (`norm_value`, default 1). We use three-class fuzzy c-means over the brain
mask of a T1-w image to get soft tissue memberships, sorted ascending by cluster center so the classes map to CSF/GM/WM,
and the tissue "mean" is the membership-weighted average. For other contrasts of the same subject, the memberships from
the co-registered T1-w image are reused.

## Kernel Density Estimation (KDE)

KDE-based normalization estimates the empirical probability density of the foreground intensities and scales the image
so the mode of the tissue of interest (white matter for T1-w) equals a constant:

$$ \hat{p}(x) = \frac{1}{n\,\delta} \sum_{i=1}^{n} K\left(\frac{x - x_i}{\delta}\right) $$

where $K$ is a Gaussian kernel, $\delta$ the bandwidth, and $n$ the number of foreground voxels (subsampled
deterministically for speed; the mode is statistically unchanged). The smooth density makes mode-finding robust. Which
local maximum is the tissue of interest depends on the modality: the **last** mode for T1-w, the **largest** for
T2-w/FLAIR, the **first** for PD/MD. If $p$ is the tissue mode,

$$ I_{\text{kde}}(\mathbf{x}) = \frac{c \cdot I(\mathbf{x})}{p}. $$

## Piecewise Linear Histogram Matching (Nyúl & Udupa)

Histogram matching [1] learns a *standard histogram* for a set of images of one contrast and maps each image's
intensities onto it, demarcated by landmark percentiles (default: $1, 10, 20, \ldots, 90, 99$; values outside $[1\%,
99\%]$ are treated as outliers) [2].

### Learning the standard histogram

Let $\mathbf{I} = \{I_1, I_2, \ldots, I_K\}$ be $K$ images of one contrast. For each image $I_i$, compute the 1% and 99%
intensity values $m_1^i, m_{99}^i$ and linearly map the image to a standard range $[m_{\text{min}}^s, m_{\text{max}}^s]$
(default $[1, 100]$):

$$ \tilde{I}_i(\mathbf{x}) = \left(I_i(\mathbf{x}) - m_1^i + m_{\text{min}}^s\right) \frac{m_{\text{max}}^s}{m_{99}^i}.
$$

Then compute the landmark percentiles of $\tilde{I}_i$ and average them across the set — the mean of each corresponding
landmark is the learned standard scale:

$$ m_n^s = \frac{1}{K} \sum_{i=1}^{K} \tilde{m}_n^i \quad \text{for } n \in \{1, 10, \ldots, 90, 99\}. $$

### Normalizing new images

For a new image, compute its landmark percentiles $\{m_1, m_{10}, \ldots, m_{99}\}$ and piecewise-linearly interpolate
each decile onto the standard scale:

$$ I_{\text{nyul}}(\mathbf{x}) = \frac{I(\mathbf{x}) - m_i}{m_j - m_i}\left(m_j^s - m_i^s\right) + m_i^s \quad \text{for
} I(\mathbf{x}) \in [m_i, m_j). $$

Because the mapping is a fixed piecewise-linear function after fitting, the saved transform applies identically to any
later image.

## WhiteStripe

WhiteStripe [3] performs a z-score-like standardization within the normal-appearing white matter (NAWM). Smooth the
foreground histogram (KDE) and take the tissue mode $\mu$ (last mode for T1-w). The "white stripe" is the intensity band
within a quantile window around the mode: with $F$ the empirical CDF of the foreground and $\tau = 5\%$ (`width`),

$$ \Omega_\tau = \left\{ I(\mathbf{x}) \mid F^{-1}\left(F(\mu) - \tau\right) < I(\mathbf{x}) < F^{-1}\left(F(\mu) +
\tau\right) \right\}. $$

Let $\hat\mu$ and $\hat\sigma$ be the mean and standard deviation over $\Omega_\tau$. Then

$$ I_{\text{ws}}(\mathbf{x}) = \frac{I(\mathbf{x}) - \hat\mu}{\hat\sigma}. $$

## Least Squares (LSQ)

LSQ pulls the CSF/GM/WM tissue means of every image toward a common standard in a least-squares sense. Tissue means are
computed with fuzzy c-means memberships (as in FCM). The standard tissue means $\mathbf{m}^s \in \mathbb{R}^3$ are
learned from a reference image (scaled so its CSF mean equals `norm_value`). Each image $I$ with tissue means
$\mathbf{m}$ is then scaled by the factor minimizing $\lVert \mathbf{m}/s - \mathbf{m}^s \rVert^2$:

$$ s = \frac{\mathbf{m}^\top \mathbf{m}}{\mathbf{m}^\top \mathbf{m}^s}, \qquad I_{\text{lsq}}(\mathbf{x}) = \frac{c
\cdot I(\mathbf{x})}{s}. $$

## RAVEL

RAVEL [4] improves on WhiteStripe by removing unwanted technical variation (e.g., scanner effects). It assumes a
population of WhiteStripe-normalized images of one contrast follows the additive model

$$ V = \alpha 1^\top + \beta X^\top + \gamma Z^\top + R $$

where $V$ is the image matrix (rows are voxels, columns are images), $\alpha 1^\top$ the average scan, $\beta X^\top$
known clinical covariates, $\gamma Z^\top$ unknown unwanted factors, and $R$ residuals. Using voxels where clinical
covariates are assumed absent — CSF control voxels — the unwanted factors are identifiable: with $V_c$ the control-voxel
matrix,

$$ V_c = \gamma Z^\top + R = U \Sigma W^\top $$

via the SVD. The first $b$ right singular vectors $W_b$ form an orthogonal basis for the unwanted factors $Z$ [5].
Voxel-wise linear regression gives the coefficients $\gamma$, and

$$ I_{\text{ravel}}(\mathbf{x}) = I_{\text{ws}}(\mathbf{x}) - \gamma_{\mathbf{x}} Z^\top. $$

Following the original paper, the default is $b = 1$; the first right singular vector is highly correlated (>95%) with
the mean CSF intensity. RAVEL requires same-shape, co-registered images and corrects the batch it was fit on — it is not
a fit-once/apply-later method.

## References

1. L. G. Nyúl, J. K. Udupa, and X. Zhang, "New Variants of a Method of MRI Scale Standardization," *IEEE Trans. Med.
   Imaging*, vol. 19, no. 2, pp. 143–150, 2000.
2. M. Shah et al., "Evaluating intensity normalization on MRIs of human brain with multiple sclerosis," *Med. Image
   Anal.*, vol. 15, no. 2, pp. 267–282, 2011.
3. R. T. Shinohara et al., "Statistical normalization techniques for magnetic resonance imaging," *NeuroImage Clin.*,
   vol. 6, pp. 9–19, 2014.
4. J. P. Fortin et al., "Removing inter-subject technical variability in magnetic resonance imaging studies,"
   *NeuroImage*, vol. 132, pp. 198–212, 2016.
5. J. T. Leek and J. D. Storey, "Capturing heterogeneity in gene expression studies by surrogate variable analysis,"
   *PLoS Genet.*, vol. 3, no. 9, pp. 1724–1735, 2007.
