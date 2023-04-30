import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from experiment.result import ExperimentResult
from simulation.config import SimulationConfig, SimulationType


def avg_results_gif(simulation_config: SimulationConfig, results: ExperimentResult, out_file_path: str, num_of_steps_in_gif: int = 25) -> None:
    num_of_repetitions = len(results.simulation_map)
    data = []
    # iteration_num == 1 or (iteration_num + 1) % self.audit_iteration_every == 0 or (iteration_num - 1) == self.num_iterations
    audited_iterations = simulation_config.audited_iterations()
    iterations_for_gif = [audited_iterations[int(i * (len(audited_iterations) / num_of_steps_in_gif))] for i in range(num_of_steps_in_gif)]
    for iteration in iterations_for_gif:
        avg_opinions_list = []
        for i in range(simulation_config.num_of_agents):
            a = []
            for repetition, simulation_result in results.simulation_map.items():
                a.append(simulation_result.get_iteration(iteration).get_opinion(i))
            avg_opinions_list.append(sum(a) / num_of_repetitions)
        # avg_opinions_list = [
        #     sum([simulation_result.get_iteration(iteration).get_opinion(i) for repetition, simulation_result in
        #          results.simulation_map]) / num_of_repetitions for i in
        #     range(simulation_config.num_of_agents)]
        # print_analysis(iteration, avg_opinions_list)
        data.append(avg_opinions_list)

    time_steps = len(data)
    fig, ax = plt.subplots()

    def animate(i):
        fig.clear()
        ax = fig.add_subplot(111, xlim=(0, 1))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)

        # Ugly code to make the title
        # Could I write it better? Yes.
        # Do I want to? No.
        # Will I? No.
        # Should I? Yes.
        if simulation_config.simulation_type == SimulationType.REPULSIVE:
            title = "Repulsion and Attraction Dynamics"
        elif simulation_config.simulation_type == SimulationType.SIMILARITY:
            title = "Only Attraction Dynamics"
        title = r'{} $\eta$={}'.format(title, simulation_config.radical_exposure_eta if simulation_config.radical_exposure_eta is not None else 0)
        if i > 0:
            ax.set_title(title)
        else:
            ax.set_title("Initial Opinion Distribution")

        ax.hist(data[i], range=(0, 1), bins=10, color='blue', alpha=0.5)
        ax.set_ylabel("number of agents")
        ax.set_xlabel("opinion")
        # df.plot.hexbin(x='opinion', y='y', gridsize=100, cmap='copper_r', norm=Normalize(0, 5, clip=True), mincnt=1, ax=ax)

    ani = animation.FuncAnimation(fig, animate, interval=1000, frames=time_steps)
    ani.save(out_file_path, writer='pillow')


def print_analysis(iteration, opinions_list):
    # Calculate the point density
    xy = np.vstack([opinions_list])
    z = gaussian_kde(xy)(xy)
    plt.scatter(opinions_list, opinions_list, c=z)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f"Opinion distribution after {int(iteration) - 1} iterations")
