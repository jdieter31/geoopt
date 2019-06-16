geoopt
======

|Python Package Index| |Read The Docs| |Build Status| |Coverage Status| |Codestyle Black| |Gitter|

Manifold aware ``pytorch.optim``.

Unofficial implementation for `“Riemannian Adaptive Optimization
Methods”`_ ICLR2019 and more.

What is done so far
-------------------

Work is in progress but you can already use this. Note that API might
change in future releases.

Tensors
~~~~~~~

-  ``geoopt.ManifoldTensor`` – just as torch.Tensor with additional
   ``manifold`` keyword argument.
-  ``geoopt.ManifoldParameter`` – same as above, recognized in
   ``torch.nn.Module.parameters`` as correctly subclassed.

All above containers have special methods to work with them as with
points on a certain manifold

-  ``.proj_()`` – inplace projection on the manifold.
-  ``.proju(u)`` – project vector ``u`` on the tangent space. You need
   to project all vectors for all methods below.
-  ``.egrad2rgrad(u)`` – project gradient ``u`` on Riemannian manifold
-  ``.inner(u, v=None)`` – inner product at this point for two
   **tangent** vectors at this point. The passed vectors are not
   projected, they are assumed to be already projected.
-  ``.retr(u, t=1.)`` – retraction map following vector ``u`` for time
   ``t``
-  ``.expmap(u, t=1.)`` – exponential map following vector ``u`` for time
   ``t`` (if expmap is not available in closed form, best approximation is used)
-  ``.transp(v, *more, u, t=1.)`` – transport vector ``v`` (and possibly
   more vectors) with direction ``u`` for time ``t``
-  ``.retr_transp(v, *more, u, t=1.)`` – transport ``self``, vector ``v``
   (and possibly more vectors) with direction ``u`` for time ``t``
   (returns are plain tensors)

Manifolds
~~~~~~~~~

-  ``geoopt.Euclidean`` – unconstrained manifold in ``R`` with
   Euclidean metric
-  ``geoopt.Stiefel`` – Stiefel manifold on matrices
   ``A in R^{n x p} : A^t A=I``, ``n >= p``
-  ``geoopt.Sphere`` - Sphere manifold ``||x||=1``
-  ``geoopt.PoincareBall`` - Poincare ball model (`wiki <https://en.wikipedia.org/wiki/Poincar%C3%A9_disk_model>`_)


All manifolds implement methods necessary to manipulate tensors on manifolds and
tangent vectors to be used in general purpose. See more in `documentation`_.

Optimizers
~~~~~~~~~~

-  ``geoopt.optim.RiemannianSGD`` – a subclass of ``torch.optim.SGD``
   with the same API
-  ``geoopt.optim.RiemannianAdam`` – a subclass of ``torch.optim.Adam``

Samplers
~~~~~~~~

-  ``geoopt.samplers.RSGLD`` – Riemannian Stochastic Gradient Langevin
   Dynamics
-  ``geoopt.samplers.RHMC`` – Riemannian Hamiltonian Monte-Carlo
-  ``geoopt.samplers.SGRHMC`` – Stochastic Gradient Riemannian
   Hamiltonian Monte-Carlo

.. _“Riemannian Adaptive Optimization Methods”: https://openreview.net/forum?id=r1eiqi09K7
.. _documentation: https://geoopt.readthedocs.io/en/latest/manifolds.html


.. |Python Package Index| image:: https://img.shields.io/pypi/v/geoopt.svg
   :target: https://pypi.python.org/pypi/geoopt
.. |Read The Docs| image:: https://readthedocs.org/projects/geoopt/badge/?version=latest
   :target: https://geoopt.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. |Build Status| image:: https://travis-ci.com/geoopt/geoopt.svg?branch=master
   :target: https://travis-ci.com/geoopt/geoopt
.. |Coverage Status| image:: https://coveralls.io/repos/github/geoopt/geoopt/badge.svg?branch=master
   :target: https://coveralls.io/github/geoopt/geoopt?branch=master
.. |Codestyle Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black
.. |Gitter| image:: https://badges.gitter.im/geoopt/community.png
   :target: https://gitter.im/geoopt/community
