About the Gradient Calculation
==============================

Matrix representation of discrete Fourier transforms
----------------------------------------------------

The discrete Fourier transform of a 1-d array :math:`x_n` is

.. math:: X_k = \sum_{n=0}^{N-1} x_n e^{-i 2 \pi k n / N}

We can write this in matrix form,

.. math::

   \mathbf{X} = \left( \begin{array}{cccc}
   e^{-i 2\pi 0 \cdot 0 / N} & e^{-i 2\pi 0\cdot 1 / N} & e^{-i 2\pi 0 \cdot 2 / N} & \cdots \\
   e^{-i 2\pi 1 \cdot 0 / N} & e^{-i 2\pi 1 \cdot1 / N} & e^{-i 2\pi 1 \cdot 2 / N} & \cdots \\
   e^{-i 2\pi 2 \cdot 0 / N} & e^{-i 2\pi 2 \cdot 1 / N} & e^{-i 2\pi 2 \cdot 2 / N} & \cdots \\
   \vdots & \vdots & \vdots & \ddots
   \end{array} \right)
   \left( \begin{array}{c} x_0 \\ x_1 \\ x_2 \\ \vdots \end{array} \right)

For a shorthand, we will call the square matrix :math:`\mathbf{F}`:

.. math:: \mathbf{X} = \mathbf{F} \mathbf{x}

Similarly, the inverse transform can be written

.. math:: x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k e^{i 2 \pi k n / N}

or

.. math::

   \mathbf{x} = \frac{1}{N} \left( \begin{array}{cccc}
   e^{i 2\pi 0 \cdot 0 / N} & e^{i 2\pi 1 \cdot 0 / N} & e^{i 2\pi 2 \cdot 0 / N} & \cdots \\
   e^{i 2\pi 0 \cdot 1 / N} & e^{i 2\pi 1 \cdot 1 / N} & e^{i 2\pi 2 \cdot 1 / N} & \cdots \\
   e^{i 2\pi 0 \cdot 2 / N} & e^{i 2\pi 1 \cdot 2 / N} & e^{i 2\pi 2 \cdot 2 / N} & \cdots \\
   \vdots & \vdots & \vdots & \ddots
   \end{array} \right)
   \left( \begin{array}{c} X_0 \\ X_1 \\ X_2 \\ \vdots \end{array} \right)

or as shorthand,

.. math:: \mathbf{x} = \mathbf{F}^{-1} \mathbf{X}

Now, note that the matrix :math:`\mathbf{F}` has the properties

.. math::

   \mathbf{F}^{-1} = \frac{1}{N} \overbar{\mathbf{F}}

where :math:`\overbar{\mathbf{F}}` is the element-wise complex
conjugate of :math:`\mathbf{F}`.
