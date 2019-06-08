import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


a = cv.imread("./img/man-m.png")
# a, _, _ = cv.split(a)
a = cv.cvtColor(a, cv.COLOR_RGB2GRAY)
a = cv.copyMakeBorder(a, 1, 1, 1, 1, cv.BORDER_REPLICATE)
a = a / 255.0

b = np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.float32)
# b = np.array([[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]], dtype=np.float32)
a_full = cv.copyMakeBorder(a, top=(b.shape[0]) // 2, bottom=(b.shape[0] + 1) // 2,
                           left=(b.shape[1]) // 2, right=(b.shape[1] + 1) // 2,
                           borderType=cv.BORDER_CONSTANT, value=0)
b_full = cv.copyMakeBorder(b, top=(a_full.shape[0] - b.shape[0]) // 2, bottom=(a_full.shape[0] - b.shape[0] + 1) // 2,
                           left=(a_full.shape[1] - b.shape[1]) // 2, right=(a_full.shape[1] - b.shape[1] + 1) // 2,
                           borderType=cv.BORDER_CONSTANT, value=0)
# b_full = cv.copyMakeBorder(b, top=0, bottom=a.shape[0], left=0, right=a.shape[1],
#                            borderType=cv.BORDER_CONSTANT, value=0)
# b_full = np.dstack([b_full, b_full, b_full])
ab = cv.filter2D(a_full, -1, b)         # b_full or b, the same
print("min:", np.min(ab), "max:", np.max(ab))
# ab = ab / 2.0 + 0.5

plt.subplot(121); plt.imshow(ab, cmap='gray'); plt.title("Convolution in time")
print("1 shape", ab.shape)

ab_f = np.fft.fft2(ab)
ab_f = np.fft.ifft2(ab_f)
ab_f = np.abs(ab_f)
# plt.subplot(132); plt.imshow(ab_f, cmap='gray'); plt.title("Der after fft&ifft")

af = np.fft.fft2(a_full)
print(af.shape)
bf = np.fft.fft2(b_full)
# bf_full = np.zeros(a.shape)
# bf_full[((a.shape[0] - b.shape[0]) // 2): (a.shape[0] - (a.shape[0] - b.shape[0]) // 2),
#         ((a.shape[1] - b.shape[1]) // 2): (a.shape[1] - (a.shape[1] - b.shape[1]) // 2), :] = np.real(bf)
# bf = cv.copyMakeBorder(bf, top=(a.shape[0] - b.shape[0]) // 2, bottom=(a.shape[0] - b.shape[0] + 1) // 2,
#                        left=(a.shape[1] - b.shape[1]) // 2, right=(a.shape[1] - b.shape[1] + 1) // 2,
#                        borderType=cv.BORDER_CONSTANT, value=0)
# print(bf.shape)
f_ab = af * bf
# f_ab = cv.mulSpectrums(af, bf, flags=cv.DFT_COMPLEX_OUTPUT)
print(f_ab.shape)
# f_ab_cut = f_ab[:a.shape[0] - b.shape[0] + 1, :a.shape[1] - b.shape[1] + 1, :]
# f_ab_cut = np.fft.ifft2(f_ab_cut)
# f_ab_cut = np.abs(f_ab_cut)
# f_ab_cut = f_ab_cut / 2.0 + 0.5

f_ab = np.fft.ifft2(f_ab)
f_ab = np.fft.fftshift(f_ab)
f_ab = np.real(f_ab)
f_ab = -f_ab
# f_ab = np.abs(f_ab)
print("min:", np.min(f_ab), "max:", np.max(f_ab))
plt.subplot(122); plt.imshow(f_ab, cmap='gray'); plt.title("Multiplication in frequency")
print("2 shape", f_ab.shape)
mx, mn = np.max(f_ab), np.min(f_ab)
# if mn < 0:
#     f_ab = f_ab - mn
# f_ab = f_ab / (mx - mn)
# plt.subplot(133); plt.imshow(f_ab, cmap='gray'); plt.title("-_-")
print("min:", np.min(f_ab), "max:", np.max(f_ab))


#
# dft_h = cv.getOptimalDFTSize(a.shape[0] + b.shape[0] - 1)
# dft_w = cv.getOptimalDFTSize(a.shape[1] + b.shape[1] - 1)
# dft_c = 3
# dft_size = [dft_h, dft_w, dft_c]
#
# tempa = np.zeros(dft_size)
# tempa[:a.shape[0], :a.shape[1]] = a
# tempb = np.zeros(dft_size)
# tempb[:b.shape[0], :b.shape[1]] = b
# c = np.zeros((abs(a.shape[0] - b.shape[0]) + 1, abs(a.shape[1] - b.shape[1]) + 1, 3))
#
# # af = cv.dft(tempa)
# af = np.fft.fft2(tempa)
# # print("f(a)", af)
# # bf = cv.dft(tempb)
# bf = np.fft.fft2(tempb)
# afbf = af * bf
# c = afbf[:c.shape[0], :c.shape[1], :]
# afbf = cv.mulSpectrums(af, bf, flags=cv.DFT_COMPLEX_OUTPUT)
# print("f(a)*f(b)", afbf)
# ab_f = np.fft.ifft2(c)
# ab_f = np.real(ab_f)
# print(ab_f)
# plt.subplot(122); plt.imshow(ab_f)


plt.show()
