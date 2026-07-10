import numpy as np
import matplotlib.pyplot as plt



def draw_earth(ax):

    u=np.linspace(0,2*np.pi,100)

    v=np.linspace(0,np.pi,100)


    x=6371*np.outer(
        np.cos(u),
        np.sin(v)
    )

    y=6371*np.outer(
        np.sin(u),
        np.sin(v)
    )

    z=6371*np.outer(
        np.ones(np.size(u)),
        np.cos(v)
    )


    ax.plot_surface(
        x,
        y,
        z,
        alpha=0.3
    )



def draw_orbit(orbits):


    fig=plt.figure(
        figsize=(8,8)
    )


    ax=fig.add_subplot(
        111,
        projection="3d"
    )


    # 画地球
    draw_earth(ax)



    # 画所有轨道
    for orbit in orbits:


        ax.plot(
            orbit[:,0],
            orbit[:,1],
            orbit[:,2]
        )



    ax.set_xlabel("X km")

    ax.set_ylabel("Y km")

    ax.set_zlabel("Z km")


    plt.show()