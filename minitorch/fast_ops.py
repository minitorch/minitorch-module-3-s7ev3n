from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # Compute total number of elements in output
        size = len(out)

        # Check if tensors are stride-aligned (same shape and strides)
        stride_aligned = (
            len(out_shape) == len(in_shape) and
            out_shape == in_shape and
            out_strides == in_strides
        )

        if stride_aligned:
            # Direct memory access without indexing
            for i in prange(size):
                out[i] = fn(in_storage[i])
        else:
            # allocate index buffers sized to each tensor's ndim
            out_index = np.zeros(len(out_shape), dtype=np.int32)
            in_index = np.zeros(len(in_shape), dtype=np.int32)

            for i in prange(size):
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)

                out_pos = index_to_position(out_index, out_strides)
                in_pos = index_to_position(in_index, in_strides)

                out[out_pos] = fn(in_storage[in_pos])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # Compute total number of elements in output
        size = len(out)

        # Check if all tensors are stride-aligned (same shape and strides)
        stride_aligned = (
            len(out_shape) == len(a_shape) == len(b_shape) and
            out_shape == a_shape == b_shape and
            out_strides == a_strides == b_strides
        )

        if stride_aligned:
            # Direct memory access without indexing
            for i in prange(size):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            # allocate index buffers sized to each tensor's ndim
            out_index = np.zeros(len(out_shape), dtype=np.int32)
            a_index = np.zeros(len(a_shape), dtype=np.int32)
            b_index = np.zeros(len(b_shape), dtype=np.int32)

            for i in prange(size):
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, a_shape, a_index)
                broadcast_index(out_index, out_shape, b_shape, b_index)

                out_pos = index_to_position(out_index, out_strides)
                a_pos = index_to_position(a_index, a_strides)
                b_pos = index_to_position(b_index, b_strides)

                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables
    为什么这里不能写入non-local variables？
    因为numba的并行化要求每个线程独立工作，不能共享状态，即prange中的循环迭代是并行执行的；
    如果在并行循环中写入non-local variables会导致数据竞争和不确定行为.
    哪些是non-local variables呢？就是main loop外部定义的变量

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # Compute total number of elements in output
        size = len(out)

        # allocate index buffers sized to each tensor's ndim
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        a_index = np.zeros(len(a_shape), dtype=np.int32)

        for i in prange(size):
            to_index(i, out_shape, out_index)

            # Copy output index to input index, but set reduced dimension to 0
            for j in range(len(out_shape)):
                a_index[j] = out_index[j]
            a_index[reduce_dim] = 0

            # Find the output position
            out_pos = index_to_position(out_index, out_strides)

            # Reduce along the specified dimension
            current = a_storage[index_to_position(a_index, a_strides)]

            # Iterate through all values in the reduced dimension
            # Pre-compute stride for the reduced dimension to avoid function calls
            reduce_stride = a_strides[reduce_dim]
            base_pos = index_to_position(a_index, a_strides)

            for k in range(1, a_shape[reduce_dim]):
                # Update position by adding stride instead of recalculating
                base_pos += reduce_stride
                current = fn(current, a_storage[base_pos])

            out[out_pos] = current

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # TODO: Implement for Task 3.2.
    raise NotImplementedError("Need to implement for Task 3.2")

def _tensor_matrix_multiply_2(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```
    这个版本实现matrix multiplication by applying a broadcasted zip and then a reduce，
    For example, consider a tensor of size (2, 4) and a tensor of size (4, 3). 
    We first zip these together with broadcasting to produce a tensor of size (2, 4, 3).
    因此，实现的时候我们创建一个临时的中间变量temp_storage来存储这个(2,4,3)的结果，
    然后再使用reduce沿着中间维度进行求和，得到最终的(2,3)结果。

    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    # Get dimensions from shapes: A(B, M, K), B(B, K, N) -> Out(B, M, N)
    B = out_shape[0]
    M = out_shape[1]   # M
    N = out_shape[2]   # N
    K = a_shape[2]    # K (a_shape[-1] == b_shape[-2])

    # Step 1: Create intermediate 4D tensor for zip operation: (B, M, K, N)
    # This will store A * B products before reduction
    temp_size = B * M * K * N
    temp_storage = np.zeros(temp_size, dtype=np.float64)
    temp_shape = np.array([B, M, K, N], dtype=np.int32)
    temp_strides = np.array([
        M * K * N,  # B stride
        K * N,      # M stride
        N,          # K stride
        1           # N stride
    ], dtype=np.int32)

    # Step 2: Broadcast A and B to 4D shapes for zip operation
    # A(B, M, K) -> (B, M, K, 1) by adding N=1 dimension
    a_4d_shape = np.array([B, M, K, 1], dtype=np.int32)
    a_4d_strides = np.array([a_strides[0], a_strides[1], a_strides[2], 0], dtype=np.int32)

    # B(B, K, N) -> (B, 1, K, N) by adding M=1 dimension
    b_4d_shape = np.array([B, 1, K, N], dtype=np.int32)
    b_4d_strides = np.array([b_strides[0], 0, b_strides[1], b_strides[2]], dtype=np.int32)

    # Step 3: Zip operation - multiply A and B with broadcasting
    # Result: (B, M, K, N) where each element is A[B,M,K] * B[B,K,N]
    zip_fn = tensor_zip(lambda x, y: x * y)
    zip_fn(temp_storage, temp_shape, temp_strides,
           a_storage, a_4d_shape, a_4d_strides,
           b_storage, b_4d_shape, b_4d_strides)

    # Step 4: Reduce operation - sum along K dimension (dimension 2)
    # Reduce (B, M, K, N) -> (B, M, N) by summing over K
    reduce_fn = tensor_reduce(lambda x, y: x + y)
    reduce_fn(out, out_shape, out_strides,
              temp_storage, temp_shape, temp_strides, 2)


# Set the main tensor_matrix_multiply to use the new implementation
_tensor_matrix_multiply = _tensor_matrix_multiply_2
tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
