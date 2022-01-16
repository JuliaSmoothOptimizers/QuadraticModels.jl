import SparseArrays.nnz

SparseArrays.nnz(M::DenseMatrix) = *(size(M)...)
SparseArrays.nnz(M::Diagonal{T, <:DenseVector{T}}) where {T} = size(M, 1)
SparseArrays.nnz(M::SymTridiagonal{T, <:DenseVector{T}}) where {T} = 2 * size(M, 1) - 1
function SparseArrays.nnz(M::Symmetric{T, <:DenseMatrix{T}}) where {T}
  n = size(M, 1)
  return n * (n + 1) / 2
end
SparseArrays.nnz(M::Symmetric{T, <:AbstractSparseMatrix{T}}) where {T} = nnz(M.data)
