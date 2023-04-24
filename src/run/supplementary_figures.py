import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps


def create_stubbornness_mu_i_by_agent_index():
    mu = 0.2
    sigma = 0.075
    num_of_agents = 101

    x = list(range(0, num_of_agents))
    dist = sps.norm(loc=mu, scale=sigma)
    y = [dist.ppf((x1 + 0.5) / num_of_agents) for x1 in x]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.scatter(x, y)
    ax1.set_xlabel('agent index (i)')
    ax1.set_ylabel(r'$\mu$')

    ax2.hist(y, bins=19)
    ax2.set_xlabel(r'$\mu$')
    ax2.set_ylabel("Number of agents")
    plt.show()


# Define the function
def radical_exposure_eta_by_opinion(x1, eta):
    y1 = 1 - abs(0.5 - x1) * eta
    if y1 < 0:
        y1 = 0
    if y1 > 1:
        y1 = 1
    return y1


def create_interaction_probability_radical_exposure_table(eta):
    data = [['Opinion of j'] + [i for i in np.arange(0, 1.1, 0.1)],
            ['Interaction Probability'] + [f'{radical_exposure_eta_by_opinion(i, eta):.2f}' for i in
                                           np.arange(0, 1.1, 0.1)]]
    fig, ax = plt.subplots()
    print(data)
    # Hide axis
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=data, loc='center', cellLoc='center', colWidths=[0.4] * len(data[0]))

    # Set table properties
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 1.5)

    # Add a title to the table
    plt.title('Opinion and Interaction Probability Table', fontsize=16)

    # Show the table
    plt.show()


def create_interaction_probability_radical_exposure_plt(etas=(2, 5)):
    # Create the plot
    fig, ax = plt.subplots()

    # Create the x values
    x = np.linspace(0, 1, 1000)

    # Create the y values using the function
    for eta in etas:
        y = [radical_exposure_eta_by_opinion(i, eta) for i in x]
        ax.plot(x, y)
    # Add a title and axis labels
    ax.set_ylabel('influence probability')
    ax.set_xlabel('opinion of j')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    # add legend and title
    if len(etas) > 1:
        # ax.set_title('Interaction Probability by Opinion of j')
        ax.legend([r'$\eta$={}'.format(eta) for eta in etas])
    # else:
    #     ax.set_title('Interaction Probability by Opinion of j for $\eta$={}'.format(etas[0]))
    # Show the plot
    plt.show()


create_interaction_probability_radical_exposure_plt(etas=(2,))
create_interaction_probability_radical_exposure_plt(etas=(.5, 2, 5, 10,))
create_stubbornness_mu_i_by_agent_index()
