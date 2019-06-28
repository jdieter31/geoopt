import torch
from . import math
from ..base import Manifold

__all__ = ["PoincareBall"]


class PoincareBall(Manifold):
    """
    Poincare ball model, see more in :doc:`/extended/poincare`

    Parameters
    ----------
    c : float|tensor
        ball negative curvature

    Notes
    -----
    It is extremely recommended to work with this manifold in double precision
    """

    ndim = 1
    reversible = False
    _default_order = 1
    name = "PoincareBall"

    def __init__(self, c=1.0):
        super().__init__()
        self.register_buffer("c", torch.as_tensor(c))

    def _check_shape(self, x, name):
        ok = x.dim() > 0
        if not ok:
            reason = "'{}' on poincare ball requires more that zero dim".format(name)
        else:
            reason = None
        return ok, reason

    def _check_point_on_manifold(self, x, atol=1e-5, rtol=1e-5):
        px = math.project(x, c=self.c)
        ok = torch.allclose(x, px, atol=atol, rtol=rtol)
        if not ok:
            reason = "'x' norm lies out of the bounds [-1/sqrt(c)+eps, 1/sqrt(c)-eps]"
        else:
            reason = None
        return ok, reason

    def _check_vector_on_tangent(self, x, u, atol=1e-5, rtol=1e-5):
        return True, None

    def _dist(self, x, y, keepdim):
        return math.dist(x, y, c=self.c, keepdim=keepdim)

    def _egrad2rgrad(self, x, u):
        return math._egrad2rgrad(x, u, c=self.c)

    def _retr(self, x, u, t, indices=None):
        # always assume u is scaled properly
        u = u*t
        if indices is not None:
            x = x.index_add_(0, indices, u)
            x.index_copy_(0, indices, math.project_in_place(x[indices], c=self.c))
        else:
            x = x.add_(u)
            math.project_in_place(x, c=self.c)

        return x

    _retr_transp_default_preference = "2y"

    def _projx(self, x):
        return math.project_in_place(x, c=self.c)

    def _proju(self, x, u):
        return u

    def _inner(self, x, u, v, keepdim):
        return math.inner(x, u, v, c=self.c, keepdim=keepdim)

    def _expmap(self, x, u, t):
        return math.project(math.expmap(x, u * t, c=self.c), c=self.c)

    def _logmap(self, x, y):
        return math.logmap(x, y, c=self.c)

    def _transp2y(self, x, v, *more, y):
        if not more:
            return math.parallel_transport(x, y, v, c=self.c)
        else:
            n = len(more) + 1
            vecs = torch.stack((v,) + more, dim=0)
            transp = math.parallel_transport(x, y, vecs, c=self.c)
            return tuple(transp[i] for i in range(n))

    def _transp_follow(self, x, v, *more, u, t):
        y = self._retr(x, u, t)
        return self._transp2y(x, v, *more, y=y)

    def _expmap_transp(self, x, v, *more, u, t):
        y = self._expmap(x, u, t)
        vs = self._transp2y(x, v, *more, y=y)
        if more:
            return (y,) + vs
        else:
            return y, vs

    def _transp_follow_expmap(self, x, v, *more, u, t):
        y = self._expmap(x, u, t)
        return self._transp2y(x, v, *more, y=y)
