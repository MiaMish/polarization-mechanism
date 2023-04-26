import numpy as np
from numpy import ndarray

from simulation.config import SimulationConfig
from simulation.simulation import Simulation


class RepulsiveSimulation(Simulation):

    def __init__(self, simulation_config: SimulationConfig):
        super().__init__(simulation_config)

    def _update_opinions(self, agent_i, agent_j) -> ndarray:
        new_opinion_i = self.opinions_list[agent_i]
        new_opinion_j = self.opinions_list[agent_j]

        delta = abs(self.opinions_list[agent_j] - self.opinions_list[agent_i])
        x = 1
        if delta > 0.5 and self.opinions_list[agent_i] < self.opinions_list[agent_j]:
            x = -1
        if delta < 0.5 and self.opinions_list[agent_i] > self.opinions_list[agent_j]:
            x = -1
        if self._is_exposed_to_passive(self.opinions_list[agent_j]):
            new_opinion_i = self._truncate_opinion(self.opinions_list[agent_i] + x * self.mio_to_use(agent_i) * delta)
        if self._is_exposed_to_passive(self.opinions_list[agent_i]):
            new_opinion_j = self._truncate_opinion( self.opinions_list[agent_j] + (-1) * x * self.mio_to_use(agent_j) * delta)
        new_opinions = np.array(self.opinions_list)
        new_opinions[agent_i] = new_opinion_i
        new_opinions[agent_j] = new_opinion_j
        return new_opinions
