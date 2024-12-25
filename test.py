import numpy as np


def im2col(input_matrix, kernel_height, kernel_width, stride = 1, padding = 0):
	# 为输入矩阵添加padding
	input_matrix_padded = np.pad(input_matrix, [(padding, padding), (padding, padding)], mode = 'constant')

	# 获取输入矩阵的尺寸
	input_height, input_width = input_matrix_padded.shape

	# 计算输出矩阵的尺寸
	output_height = (input_height - kernel_height) // stride + 1
	output_width = (input_width - kernel_width) // stride + 1

	# 创建im2col矩阵
	cols = np.zeros((kernel_height * kernel_width, output_height * output_width))

	# 填充im2col矩阵
	for i in range(output_height):
		for j in range(output_width):
			region = input_matrix_padded[i * stride:i * stride + kernel_height, j * stride:j * stride + kernel_width]
			cols[:, i * output_width + j] = region.flatten()

	return cols


def conv2d_im2col(input_matrix, kernel, stride = 1, padding = 0):
	kernel_height, kernel_width = kernel.shape
	# 1. 使用 im2col 展开输入矩阵
	input_cols = im2col(input_matrix, kernel_height, kernel_width, stride, padding)

	# 2. 将卷积核展开成列向量
	kernel_cols = kernel.flatten()[:, np.newaxis]

	# 3. 进行矩阵乘法，得到卷积结果
	output_cols = np.dot(kernel_cols.T, input_cols)

	# 4. 重塑输出矩阵为适当的形状
	output_height = (input_matrix.shape[0] - kernel_height) // stride + 1
	output_width = (input_matrix.shape[1] - kernel_width) // stride + 1
	output_matrix = output_cols.reshape(output_height, output_width)

	return output_matrix


# 示例：对输入矩阵进行卷积操作
input_matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype = float)
kernel = np.array([[1, 0], [0, -1]], dtype = float)

output_matrix = conv2d_im2col(input_matrix, kernel)
print(output_matrix)