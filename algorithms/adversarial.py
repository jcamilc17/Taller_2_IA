from abc import ABC, abstractmethod

from algorithms.evaluation import evaluation_function
from world.game_state import GameState


class MultiAgentSearchAgent(ABC):
    """Clase base para los agentes de búsqueda adversaria."""

    def __init__(self, depth: int | str = 2) -> None:
        self.depth = int(depth)
        if self.depth < 1:
            raise ValueError("La profundidad debe ser al menos 1 ply")
        self.nodes_evaluated = 0

    @abstractmethod
    def get_action(self, state: GameState) -> str | None:
        raise NotImplementedError


class MinimaxAgent(MultiAgentSearchAgent):
    """Agente Minimax para el defensor MAX frente al intruso MIN."""

    def get_action(self, state: GameState) -> str | None:
        """
        Retorna la acción del defensor con mayor valor Minimax.

        El defensor es MAX (agente 0), el intruso es MIN (agente 1) y cada
        acción consume un ply. Debe respetar el orden de las acciones legales,
        usar evaluation_function en terminales y cortes, y contar cada estado
        procesado una vez en self.nodes_evaluated, incluida la raíz.

        Tips:
        - Use state.get_legal_actions(agent_index) y
          state.generate_successor(agent_index, action) para expandir el árbol.
        - Compruebe state.is_win(), state.is_lose() y el corte de profundidad;
          evalúe esos estados con evaluation_function(state).
        - El siguiente agente es (agent_index + 1) % state.get_num_agents().
          depth=1 incluye una acción de MAX y depth=2 una de MAX y una de MIN.
        - Reinicie las métricas y cuente una vez cada estado procesado, incluida
          la raíz. Retorne la acción de MAX y conserve la primera en los empates.
        """
        # TODO: Add your code here
        raise NotImplementedError("Punto 4: implemente MinimaxAgent.get_action")


class AlphaBetaAgent(MultiAgentSearchAgent):
    """Agente Minimax que evita explorar ramas mediante poda alfa-beta."""

    def get_action(self, state: GameState) -> str | None:
        """
        Retorna la acción de Minimax aplicando poda alfa-beta.

        Debe usar la misma profundidad, orden de acciones y función de
        evaluación que Minimax.

        Tips:
        - Conserve la misma estructura y casos base de MinimaxAgent.
        - Inicie alpha en -infinito y beta en +infinito, y páselos en las
          llamadas recursivas.
        - En MAX actualice alpha y corte si valor >= beta; en MIN actualice beta
          y corte si valor <= alpha.
        """
        self.nodes_evaluated = 0
        num_agents = state.get_num_agents()

        def value(node: GameState, agent_index: int, depth_left: int, alpha: float, beta: float) -> float:
            self.nodes_evaluated += 1
            if node.is_win() or node.is_lose() or depth_left == 0:
                return evaluation_function(node)
            next_agent = (agent_index + 1) % num_agents
            if agent_index == 0:
                best = float("-inf")
                for action in node.get_legal_actions(agent_index):
                    successor = node.generate_successor(agent_index, action)
                    best = max(best, value(successor, next_agent, depth_left - 1, alpha, beta))
                    if best >= beta:
                        return best
                    alpha = max(alpha, best)
                return best
            best = float("inf")
            for action in node.get_legal_actions(agent_index):
                successor = node.generate_successor(agent_index, action)
                best = min(best, value(successor, next_agent, depth_left - 1, alpha, beta))
                if best <= alpha:
                    return best
                beta = min(beta, best)
            return best

        self.nodes_evaluated += 1
        best_action: str | None = None
        best_value = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        for action in state.get_legal_actions(0):
            successor = state.generate_successor(0, action)
            action_value = value(successor, 1 % num_agents, self.depth - 1, alpha, beta)
            if action_value > best_value:
                best_value = action_value
                best_action = action
            alpha = max(alpha, best_value)
        return best_action
