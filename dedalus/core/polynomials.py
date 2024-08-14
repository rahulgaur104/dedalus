import jax
import jax.numpy as jnp

@jax.jit
def chebyshev_derivative_2d_jax(A):
    I, J = A.shape
    n = jnp.arange(J)
    
    # Compute the derivative coefficients
    B = jnp.zeros_like(A)
    B = B.at[:, 1:].set(2 * n[1:, None] * A[:, 1:].T).T
    
    # Cumulative sum to compute the final coefficients
    B = jnp.flip(jnp.cumsum(jnp.flip(B[:, 1:], axis=1), axis=1), axis=1)
    
    # Handle the first coefficient separately
    B = B.at[:, 0].set(A[:, 1] + 0.5 * B[:, 1])
    
    return B

@jax.jit
def legendre_derivative_2d_jax(A):
    I, J = A.shape
    n = jnp.arange(J)
    
    # Compute the derivative coefficients
    B = jnp.zeros_like(A)
    B = B.at[:, :-1].set((2 * n[1:, None] - 1) * A[:, 1:].T).T
    
    # Compute the cumulative factor
    factor = jnp.cumprod(2 * n + 1) / (2 * n + 1)
    factor = jnp.flip(factor[1:])
    
    # Apply the cumulative factor
    B = B[:, :-1] * factor[None, :]
    
    # Cumulative sum to compute the final coefficients
    B = jnp.flip(jnp.cumsum(jnp.flip(B, axis=1), axis=1), axis=1)
    
    return B

