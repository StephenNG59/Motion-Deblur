kernel_size = (11, 11)
window_size = kernel_size
# smooth_threshold = 0.05
smooth_threshold = 5
# file_path = "./img/asc-s.jpg"
file_path = "./img/man-s.png"
mask_path = "./img/mask-m.bmp"
mask_np = "./img/mask.npy"
psi_updated_path = "./img/psi_updated.npy"

Gamma = 1
Lambda1 = 0.002
Lambda2 = 10

LatentThreshold = 1e-5
PsiThreshold = 1e-5
