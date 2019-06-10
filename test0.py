import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


time_1 = True
is_gray = False
rgb_scale = 2
a = cv.imread("./img/man-m.png")
# a, _, _ = cv.split(a)
if is_gray:
    a = cv.cvtColor(a, cv.COLOR_RGB2GRAY)
a = a / 255.0
# a = cv.copyMakeBorder(a, 1, 1, 1, 1, cv.BORDER_REPLICATE)

# b = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.float32)
b = np.array([[-1, 1, -1], [-1, 4, -1], [-1, 1, -1]], dtype=np.float32)


# b = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], dtype=np.float32)
a_full = cv.copyMakeBorder(a, top=(b.shape[0]) // 2, bottom=(b.shape[0] + 1) // 2,
                           left=(b.shape[1]) // 2, right=(b.shape[1] + 1) // 2,
                           borderType=cv.BORDER_CONSTANT, value=0)
b_full = cv.copyMakeBorder(b, top=(a_full.shape[0] - b.shape[0]) // 2, bottom=(a_full.shape[0] - b.shape[0]) // 2,
                           left=(a_full.shape[1] - b.shape[1]) // 2, right=(a_full.shape[1] - b.shape[1]) // 2,
                           borderType=cv.BORDER_CONSTANT, value=0)
# b_full = cv.copyMakeBorder(b, top=0, bottom=a.shape[0], left=0, right=a.shape[1],
#                            borderType=cv.BORDER_CONSTANT, value=0)
# b_full = np.dstack([b_full, b_full, b_full])
if is_gray:
    ab = cv.filter2D(a_full, -1, b)         # b_full or b, the same
else:
    ar, ag, ab = cv.split(a_full)
    rb, gb, bb = cv.filter2D(ar, -1, b), cv.filter2D(ag, -1, b), cv.filter2D(ab, -1, b)
    ab = cv.merge([rb, gb, bb])
print("min:", np.min(ab), "max:", np.max(ab))
# ab = ab / 2.0 + 0.5

# for i in range(b.shape[0]):
#     for j in range(b.shape[1]):
#         b[i][j] = pow(-1, i+j) * b[i][j]
a_full = cv.copyMakeBorder(a, top=0, bottom=b.shape[0] - 1, left=0, right=b.shape[1] - 1,
                           borderType=cv.BORDER_CONSTANT, value=0)
if time_1:
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            a_full[i][j] = pow(-1, i+j) * a_full[i][j]
# if not is_gray:
#     b = np.dstack([b, b, b])
b_full = cv.copyMakeBorder(b, top=0, bottom=a_full.shape[0] - b.shape[0],
                           left=0, right=a_full.shape[1] - b.shape[1],
                           borderType=cv.BORDER_CONSTANT, value=0)
# ab_32f = np.ndarray(ab.shape, np.float32)
# ab_32f[:, :] = ab * 255
# ab_32f = cv.cvtColor(ab_32f, code=cv.COLOR_GRAY2RGB)
if is_gray:
    plt.subplot(121); plt.imshow(ab, cmap='gray'); plt.title("Convolution in time")
else:
    plt.subplot(121); plt.imshow(ab * rgb_scale); plt.title("Convolution in time")

# plt.subplot(121); plt.imshow(ab_32f); plt.title("Convolution in time")
print("1 shape", ab.shape)

ab_f = np.fft.fft2(ab)
ab_f = np.fft.ifft2(ab_f)
ab_f = np.abs(ab_f)
# plt.subplot(132); plt.imshow(ab_f, cmap='gray'); plt.title("Der after fft&ifft")

bf = np.fft.fft2(b_full)
ar, ag, ab = cv.split(a_full)
if is_gray:
    af = np.fft.fft2(a_full)
    print(af.shape)
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
    # f_ab = np.fft.fftshift(f_ab)
    f_ab = np.real(f_ab)
else:
    arf, agf, abf = np.fft.fft2(ar), np.fft.fft2(ag), np.fft.fft2(ab)
    f_abr, f_abg, f_abb = np.real(np.fft.ifft2(arf * bf)), np.real(np.fft.ifft2(agf * bf)), np.real(np.fft.ifft2(abf * bf))
    print("f_abr shape", f_abr.shape)
    f_ab = np.dstack([f_abr, f_abg, f_abb])

if time_1:
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            f_ab[i][j] = pow(-1, i+j) * f_ab[i][j]
# f_ab = -f_ab
# f_ab = np.abs(f_ab)
print("min:", np.min(f_ab), "max:", np.max(f_ab))
mx, mn = np.max(f_ab), np.min(f_ab)
# if mn < 0:
#     f_ab = f_ab - mn
# f_ab = f_ab / (mx - mn)
# f_ab_32f = np.ndarray(f_ab.shape, np.float32)
# f_ab_32f[:, :] = f_ab
# f_ab_32f = cv.cvtColor(f_ab_32f, code=cv.COLOR_GRAY2RGB)
if is_gray:
    plt.subplot(122); plt.imshow(f_ab, cmap='gray'); plt.title("Multiplication in frequency")
else:
    # f_ab = f_ab * 255
    plt.subplot(122); plt.imshow(f_ab * rgb_scale); plt.title("Multiplication in frequency")

# plt.subplot(122); plt.imshow(f_ab_32f); plt.title("Multiplication in frequency")
print("2 shape", f_ab.shape)
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

ab_cover = f_ab / f
