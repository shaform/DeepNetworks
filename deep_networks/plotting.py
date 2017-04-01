import sns


def plot_gaussian(ax, data, size=5):
    ax.set_aspect('equal')
    ax.set_ylim((-size, size))
    ax.set_xlim((-size, size))
    ax.tick_params(labelsize=10)
    sns.kdeplot(
        data[:, 0],
        data[:, 1],
        cmap='Blues',
        shade=True,
        shade_lowest=False,
        ax=ax)
    ax.scatter(data[:, 0], data[:, 1], linewidth=1, marker='+', color='w')
