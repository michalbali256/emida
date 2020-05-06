#!/usr/bin/python3

import numpy as np

def hexplot1(pos, C, step=None, cmap=None, vmin=None, vmax=None, ax=None, **kwargs):
    if step is None:
        step = pos[1,0]-pos[0,0]

    s3 = 1/np.sqrt(3)
    hx = np.array([(0,  2*s3), ( 1,  s3), ( 1, -s3),
                   (0, -2*s3), (-1, -s3), (-1,  s3)])*step/2

    pos6 = pos[:,None,:] + hx[None,:,:]

    kwargs.setdefault('edgecolors', 'none')
    kwargs.setdefault('antialiaseds', True)

    from matplotlib.collections import PolyCollection
    col = PolyCollection(pos6, **kwargs)
    col.set_array(C)
    col.set_cmap(cmap)
    if vmin is not None or vmax is not None:
        col.set_clim(vmin, vmax)
    else:
        col.autoscale_None()

    if ax is None:
        from matplotlib.pyplot import gca
        ax = gca()

    ax.grid(False)

    minx = pos6[...,0].min()
    maxx = pos6[...,0].max()
    miny = pos6[...,1].min()
    maxy = pos6[...,1].max()
    ax.update_datalim([ (minx, miny), (maxx, maxy) ])
    ax.margins(0)
    #ax.autoscale_view()
    ax.add_collection(col)
    ax.set_aspect("equal")
    return col

def hexplot2(m, n, step, C, ax=None, **kwargs):
    assert len(C) == m//2*(2*n-1) + m%2*n

    xx, yy = np.meshgrid(
        np.arange(-1, 2*n)*step/2,
        np.arange(-.5, m)*step/2*np.sqrt(3)
    )

    s = step/12*np.sqrt(3)
    yy[::2,::2] += s
    yy[::2,1::2] -= s
    yy[1::2,::2] -= s
    yy[1::2,1::2] += s

    s = m//2*(2*n-1)
    end = C[s:]
    C = C[:s].reshape(m//2, 2*n-1)
    even = C[:,:n]
    odd = C[:,n:]

    cc = np.zeros((m, 2*n))
    if m % 2:
        cc[:-1:2,::2] = cc[:-1:2,1::2] = even
        cc[-1,::2] = cc[-1,1::2] = end
    else:
        cc[::2,::2] = cc[::2,1::2] = even

    cc[1::2,0] = cc[1::2,-1] = np.nan
    cc[1::2,1:-1:2] = cc[1::2,2:-1:2] = odd

    if ax is None:
        from matplotlib.pyplot import gca
        ax = gca()

    mesh = ax.pcolormesh(xx, yy, cc, shading='flat', **kwargs)
    ax.set_aspect("equal")
    return mesh


if __name__ == "__main__":
    from matplotlib.pyplot import subplot, show, plot, savefig

    x, y, c = np.loadtxt("../../Testing data/FeAl/DEFORMED_FeAl.ang",usecols=(3,4,5),unpack=True)

    subplot(121)
    hexplot1(np.column_stack([x,y]), c)
    subplot(122)
    hexplot2(135, 117, 0.6, c)

    #plot(x, y, "r,")
    #savefig("test.png", dpi=1000)
    #savefig("test.pdf")

    show()
