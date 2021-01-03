#!/usr/bin/python3

import numpy as np


def add_collection(collection_class, *args, **kwargs):
    C = kwargs.pop("C")
    corners = kwargs.pop("corners")

    ax = kwargs.pop("ax", None)

    alpha = kwargs.pop("alpha", None)
    cmap = kwargs.pop("cmap", None)
    norm = kwargs.pop("norm", None)
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)

    col = collection_class(*args, **kwargs)
    col.set_alpha(alpha)
    col.set_array(C)
    col.set_cmap(cmap)
    col.set_norm(norm)
    col.set_clim(vmin, vmax)
    col.autoscale_None()

    if ax is None:
        from matplotlib.pyplot import gca
        ax = gca()

    ax.grid(False)
    ax.add_collection(col, autolim=False)
    ax.update_datalim(corners)
    ax.margins(0)
    ax.set_aspect("equal")
    return col


def hexplot1(XY, C, step=None, **kwargs):
    if step is None:
        step = XY[1,0]-XY[0,0]

    s3 = 1/np.sqrt(3)
    hx = np.array([(0,  2*s3), ( 1,  s3), ( 1, -s3),
                   (0, -2*s3), (-1, -s3), (-1,  s3)])*step/2

    pos6 = XY[:,None,:] + hx[None,:,:]

    corners = [ pos6.min(axis=(0,1)),
                pos6.max(axis=(0,1)) ]

    kwargs.setdefault('edgecolors', 'none')
    kwargs.setdefault('antialiased', False)

    from matplotlib.collections import PolyCollection
    return add_collection(PolyCollection, pos6, C=C, corners=corners, **kwargs)


def hexplot2(m, n, step, C, **kwargs):
    assert len(C) == m//2*(2*n-1) + m%2*n

    x, y = np.meshgrid(
        np.arange(-1, 2*n)*step/2,
        np.arange(-.5, m)*step/2*np.sqrt(3)
    )

    s = step/12*np.sqrt(3)
    y[::2,::2] += s
    y[::2,1::2] -= s
    y[1::2,::2] -= s
    y[1::2,1::2] += s

    s = m//2*(2*n-1)
    end = C[s:]
    C = C[:s].reshape(m//2, 2*n-1)
    even = C[:,:n]
    odd = C[:,n:]

    c = np.zeros((m, 2*n))
    if m % 2:
        c[:-1:2,::2] = c[:-1:2,1::2] = even
        c[-1,::2] = c[-1,1::2] = end
    else:
        c[::2,::2] = c[::2,1::2] = even

    c[1::2,0] = c[1::2,-1] = np.nan
    c[1::2,1:-1:2] = c[1::2,2:-1:2] = odd

    pos = np.column_stack([x.ravel(), y.ravel()])

    corners = [ pos.min(axis=(0)),
                pos.max(axis=(0)) ]

    #kwargs.setdefault('shading', 'flat')
    kwargs.setdefault('edgecolors', 'none')
    kwargs.setdefault('antialiased', False)
    kwargs.setdefault('linewidths', 0)

    from matplotlib.collections import QuadMesh
    from matplotlib.cbook import safe_masked_invalid # needed !
    return add_collection(QuadMesh, 2*n, m, pos, C=safe_masked_invalid(c).ravel(), corners=corners, **kwargs)


def hexplot3(X, Y, C, **kwargs):
    from matplotlib.tri import Triangulation
    tri = Triangulation(X,Y) # TODO: explicit triangles?

    corners = [ (X.min(), Y.min()),
                (X.max(), Y.max()) ]

    kwargs.setdefault('edgecolors', 'none')
    kwargs.setdefault('antialiaseds', False)

    from matplotlib.collections import TriMesh
    return add_collection(TriMesh, tri, C=C, corners=corners, **kwargs)


if __name__ == "__main__":
    from matplotlib.pyplot import figure, show, plot, savefig

    x, y, c = data = np.loadtxt("../../Testing data/FeAl/DEFORMED_FeAl.ang", usecols=(3,4,5), unpack=True)

    figure()
    hexplot1(data[:2].T, c)
    #savefig("hexplot1.pdf")
    savefig("hexplot1.png", dpi=500)

    figure()
    hexplot2(135, 117, 0.6, c)
    #savefig("hexplot2.pdf")
    savefig("hexplot2.png", dpi=500)

    figure()
    hexplot3(x, y, c)
    #savefig("hexplot3.pdf")
    savefig("hexplot3.png")

    #plot(x, y, "r,")

    show()
