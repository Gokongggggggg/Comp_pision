# Overview

This project evaluates the robustness of SIFT feature matching between an original image and three augmented versions: Gaussian blur, Gaussian noise, and JPEG compression. For this sample, the number of good matches after applying the Lowe Ratio Test was 26 for Gaussian blur, 20 for Gaussian noise, and 31 for JPEG compression. These results suggest that different image distortions affect SIFT keypoints and descriptors in different ways.

# Result Analysis

Gaussian noise produced the lowest number of good matches, with 20 matches. This indicates that random pixel-level intensity changes can strongly affect local image gradients. Since SIFT relies on stable gradient patterns around keypoints, noise may create unstable local structures or alter descriptor values enough to reduce reliable matching. For this sample, Gaussian noise appears to be the most disruptive augmentation among the three tested conditions.

JPEG compression preserved the highest number of good matches, with 31 matches. This suggests that although JPEG compression introduces artifacts and may remove some fine image details, many larger-scale structures remain recognizable. SIFT descriptors are designed to tolerate moderate changes in illumination, scale, and local appearance, so compression artifacts may not fully destroy the gradient patterns needed for matching. For this sample, the original and JPEG-compressed images still retained many corresponding local features.

Gaussian blur resulted in 26 good matches, which is lower than JPEG compression but higher than Gaussian noise. Blur smooths the image and reduces high-frequency detail, which can weaken corners, edges, and textured regions where SIFT keypoints are often detected. As a result, some keypoints may disappear or become less distinctive. However, broader structures can remain visible, allowing a moderate number of descriptors to still match successfully.

# Interpretation

The results indicate that SIFT descriptors have some robustness to image distortion, but this robustness depends on the type of distortion. Distortions that preserve the main spatial structure of the image, such as JPEG compression in this sample, may still allow many descriptors to match. In contrast, distortions that modify local gradient patterns more randomly, such as Gaussian noise, can reduce descriptor consistency and decrease the number of reliable matches.

This relationship shows that feature matching performance is closely connected to how much an augmentation changes local image structure. SIFT can remain effective when important edges and shapes are preserved, but matching quality may decline when keypoints become unstable or descriptors become less distinctive.

# Conclusion

For this sample, JPEG compression produced the most good matches, Gaussian blur produced a moderate number, and Gaussian noise produced the fewest. These findings suggest that SIFT is relatively robust to compression and moderate smoothing, but less robust to random noise that affects local gradients. The results should be interpreted carefully because they are based on one selected image pair, but they provide useful evidence for understanding how image distortion influences feature matching.
