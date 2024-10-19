import numpy as np
from scipy.sparse import csr_matrix

np.random.seed(40)

# Function to generate a sparse matrix with integer values 0 to 9
def generate_sparse_matrix(rows, cols, sparsity):
    num_elements = int(rows * cols * sparsity)
    data = np.random.randint(1, 10, size=num_elements)
    row_indices = np.random.randint(0, rows, size=num_elements)
    col_indices = np.random.randint(0, cols, size=num_elements)
    return csr_matrix((data, (row_indices, col_indices)), shape=(rows, cols))

# Generate sparse matrices
m, h, n = 50000, 20000, 30000
sparsity = 0.001
A = generate_sparse_matrix(m, h, sparsity)
B = generate_sparse_matrix(h, n, sparsity)

# Perform matrix multiplication
result = A.dot(B)

# Function to write sorted non-zero elements of a sparse matrix
def write_sorted_nonzero(matrix, file):
    # Extract non-zero indices and data
    row_indices, col_indices = matrix.nonzero()
    data = matrix.data
    
    # Sort by row index first and then by column index
    sorted_indices = np.lexsort((col_indices, row_indices))
    
    # Write number of non-zero elements
    file.write(f"{matrix.nnz}\n")
    
    # Write sorted non-zero elements
    for idx in sorted_indices:
        i = row_indices[idx]
        j = col_indices[idx]
        value = data[idx]
        file.write(f"{i} {j} {value}\n")
    file.write('\n')

# Format and save matrices to a text file
with open('spgemm_input_large1_0.01.txt', 'w') as f:
    # Write dimensions of the matrices
    f.write(f"{m} {h} {n}\n\n")

    # Write matrix A with sorted non-zero elements
    write_sorted_nonzero(A, f)

    # Write matrix B with sorted non-zero elements
    write_sorted_nonzero(B, f)

    # Write result matrix with sorted non-zero elements
    write_sorted_nonzero(result, f)