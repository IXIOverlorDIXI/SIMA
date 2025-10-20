import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import time
import json
import argparse
import sys

class ACOGraphColoring:
    def __init__(self, graph, num_ants, alpha, beta, rho, q, k, node_selection_strategy='dsatur'):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.num_nodes = len(self.nodes)
        self.adj = self.graph.adj
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.k = k
        self.pheromone = np.ones((self.num_nodes, self.k))
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.history = []
        self.node_selection_strategy = node_selection_strategy

    def _get_dsatur_node(self, current_coloring):
        uncolored_nodes = [node for node in self.nodes if node not in current_coloring]
        if not uncolored_nodes: return None
        saturation_degrees = {node: len({current_coloring[neighbor] for neighbor in self.adj[node] if neighbor in current_coloring}) for node in uncolored_nodes}
        max_saturation = max(saturation_degrees.values())
        candidate_nodes = [node for node, sat in saturation_degrees.items() if sat == max_saturation]
        if len(candidate_nodes) > 1:
            return max(candidate_nodes, key=lambda node: self.graph.degree(node))
        return candidate_nodes[0]

    def _get_random_node(self, current_coloring):
        uncolored_nodes = [node for node in self.nodes if node not in current_coloring]
        return random.choice(uncolored_nodes) if uncolored_nodes else None

    def _get_next_node(self, coloring):
        if self.node_selection_strategy == 'dsatur':
            return self._get_dsatur_node(coloring)
        elif self.node_selection_strategy == 'random':
            return self._get_random_node(coloring)
        else:
            raise ValueError("Unknown node selection strategy")

    def _calculate_heuristic_and_conflicts(self, node, color, coloring):
        conflicts = sum(1 for neighbor in self.adj[node] if neighbor in coloring and coloring[neighbor] == color)
        heuristic = 1.0 / (1.0 + conflicts)
        return heuristic

    def _choose_color(self, node, coloring):
        node_idx = self.node_to_idx[node]
        pheromones = self.pheromone[node_idx, :]
        heuristics = np.array([self._calculate_heuristic_and_conflicts(node, c, coloring) for c in range(self.k)])
        probabilities = (pheromones ** self.alpha) * (heuristics ** self.beta)
        sum_probs = np.sum(probabilities)
        if sum_probs == 0: return random.randint(0, self.k - 1)
        probabilities /= sum_probs
        return np.random.choice(self.k, p=probabilities)

    def _construct_solution(self, ant_id, iteration_id):
        start_time = time.perf_counter()
        coloring = {}
        strategy_text = "DSATUR" if self.node_selection_strategy == 'dsatur' else "випадковим чином"
        
        for step in range(self.num_nodes):
            current_node = self._get_next_node(coloring)
            if current_node is None: break
            
            desc_prefix = f"[Спроба для k={self.k}] "
            self.history.append({'iteration': iteration_id, 'ant': ant_id, 'type': 'node_selection', 'current_node': current_node, 'coloring': dict(coloring), 'description': desc_prefix + f"Ітерація {iteration_id+1}, Мураха {ant_id+1}:\nВершина '{current_node}' обрана {strategy_text}."})
            
            chosen_color = self._choose_color(current_node, coloring)
            coloring[current_node] = chosen_color

            conflicts = sum(1 for neighbor in self.adj[current_node] if neighbor in coloring and coloring[neighbor] == chosen_color)
            desc = desc_prefix + f"Ітерація {iteration_id+1}, Мураха {ant_id+1}:\nВершині '{current_node}' призначено колір {chosen_color+1}."
            if conflicts > 0: desc += f"\nУвага, створено конфлікт!"
            self.history.append({'iteration': iteration_id, 'ant': ant_id, 'type': 'color_assignment', 'current_node': current_node, 'coloring': dict(coloring), 'description': desc})

        total_conflicts = sum(1 for u, v in self.graph.edges() if u in coloring and v in coloring and coloring[u] == coloring[v])
        num_colors_used = len(set(coloring.values()))
        duration = time.perf_counter() - start_time
        self.history.append({'iteration': iteration_id, 'ant': ant_id, 'type': 'ant_done', 'current_node': None, 'coloring': dict(coloring), 'description': f"[Спроба для k={self.k}] Мураха {ant_id+1} завершила за {duration:.4f} сек.\nРезультат: {total_conflicts} конфліктів, {num_colors_used} кольорів."})
        return coloring, total_conflicts

    def _global_pheromone_update(self, best_solution_coloring, best_solution_conflicts, iter_duration):
        self.pheromone *= (1 - self.rho)
        deposit_amount = self.q / (1 + best_solution_conflicts)
        for node, color in best_solution_coloring.items():
            node_idx = self.node_to_idx[node]
            if color < self.k: self.pheromone[node_idx, color] += deposit_amount

        num_colors_used = len(set(best_solution_coloring.values()))
        desc = (f"[Спроба для k={self.k}] Кінець ітерації {self.history[-1]['iteration']+1} (час: {iter_duration:.4f} сек).\n"
                f"Найкраще рішення ітерації: {best_solution_conflicts} конфліктів, використано {num_colors_used} кольорів.")
        self.history.append({'iteration': self.history[-1]['iteration'], 'ant': None, 'type': 'global_update', 'current_node': None, 'coloring': best_solution_coloring, 'description': desc})

    def run(self, num_iterations=20):
        min_conflicts = float('inf')
        best_overall_coloring = None
        for i in range(num_iterations):
            iter_start_time = time.perf_counter()
            solutions_in_iteration = [self._construct_solution(j, i) for j in range(self.num_ants)]
            best_iter_coloring, best_iter_conflicts = min(solutions_in_iteration, key=lambda x: x[1])
            if best_iter_conflicts < min_conflicts:
                min_conflicts = best_iter_conflicts
                best_overall_coloring = best_iter_coloring

            if min_conflicts == 0:
                break
            
            iter_duration = time.perf_counter() - iter_start_time
            self._global_pheromone_update(best_iter_coloring, best_iter_conflicts, iter_duration)

        return best_overall_coloring, min_conflicts, self.history

class ChromaticSolver:
    def __init__(self, graph, num_ants, alpha, beta, rho, q, node_selection_strategy='dsatur'):
        self.graph = graph
        self.aco_params = {'num_ants': num_ants, 'alpha': alpha, 'beta': beta, 'rho': rho, 'q': q}
        self.node_selection_strategy = node_selection_strategy
        self.strategy_name = f"ACO + {'DSATUR' if node_selection_strategy == 'dsatur' else 'Random'}"
        
    def solve(self, start_k=None, iterations_per_k=50):
        total_start_time = time.perf_counter()
        if start_k is None:
            k = max(dict(self.graph.degree()).values()) + 1
        else:
            k = start_k

        max_clique, w = nx.algorithms.clique.max_weight_clique(self.graph, weight=None)

        clique_number = len(max_clique)

        if k < clique_number:
            print(f"WARNING: Початкове значення k={k} є меншим за розмір кліки ({clique_number}).")
            print(f"         Примусово починаємо пошук з k={clique_number}.")
            k = clique_number

        best_k_solution = None
        best_k = float('inf')
        combined_history = []
        

        checked_k_values = set()

        max_k_limit = self.graph.number_of_nodes() + 1 

        while k >= clique_number and k < max_k_limit:
            print(f"\n--- [{self.strategy_name}] Спроба знайти рішення для k = {k} кольорів ---")

            if k in checked_k_values:
                print(f"INFO: Значення k={k} вже було перевірено раніше. Завершуємо пошук, оскільки знайдено мінімальне k.")
                break
            
            aco_solver = ACOGraphColoring(graph=self.graph, **self.aco_params, k=k, node_selection_strategy=self.node_selection_strategy)
            coloring, conflicts, history = aco_solver.run(num_iterations=iterations_per_k)
            
            checked_k_values.add(k)

            combined_history.extend(history)
            
            if conflicts == 0:
                print(f"+++ [{self.strategy_name}] УСПІХ! Знайдено рішення без конфліктів для k = {k} +++")
                best_k = k
                best_k_solution = coloring
                num_colors_used = len(set(best_k_solution.values()))
                desc = (f"РІШЕННЯ ДЛЯ k={k} ЗНАЙДЕНО!\n"
                        f"Використано {num_colors_used} кольорів. Зменшуємо k і шукаємо далі...")
                combined_history.append({'iteration': -1, 'ant': None, 'type': 'solution_found', 'current_node': None, 'coloring': best_k_solution, 'description': desc})
                
                if k == clique_number:
                    print(f"INFO: Досягнуто теоретичний мінімум кольорів ({clique_number}). Пошук завершено.")
                    break

                k -= 1
            else:
                if best_k_solution is None:
                    print(f"--- [{self.strategy_name}] НЕВДАЧА для k = {k}. Збільшуємо кількість кольорів і пробуємо знову. ---")
                    desc_increase = f"Не вдалося знайти рішення для k={k}.\nЗбільшуємо k до {k+1} і повторюємо пошук."
                    combined_history.append({'iteration': -1, 'ant': None, 'type': 'k_increase', 'current_node': None, 'coloring': {}, 'description': desc_increase})
                    k += 1
                    continue
                else:
                    print(f"--- [{self.strategy_name}] НЕВДАЧА. Не вдалося знайти рішення для k = {k} за {iterations_per_k} ітерацій. ---")
                    desc = (f"ПОШУК ЗАВЕРШЕНО.\n"
                            f"Не вдалося знайти рішення для k={k}. Найкращий результат: {best_k} кольорів.")
                    combined_history.append({'iteration': -1, 'ant': None, 'type': 'search_end', 'current_node': None, 'coloring': best_k_solution if best_k_solution else {}, 'description': desc})
                    break
        
        total_duration = time.perf_counter() - total_start_time
        if best_k_solution:
            print(f"\n\n=== [{self.strategy_name}] Найкращий результат: розфарбування з {best_k} кольорами. Загальний час: {total_duration:.2f} сек. ===")
        else:
            print(f"\n\n=== [{self.strategy_name}] Не вдалося знайти рішення без конфліктів. Загальний час: {total_duration:.2f} сек. ===")
            
        return best_k_solution, best_k, combined_history

class GreedyDSATUR:
    def __init__(self, graph):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.adj = self.graph.adj
        self.history = []

    def _get_dsatur_node(self, coloring):
        uncolored_nodes = [node for node in self.nodes if node not in coloring]
        if not uncolored_nodes: return None
        saturation_degrees = {node: len({coloring[neighbor] for neighbor in self.adj[node] if neighbor in coloring}) for node in uncolored_nodes}
        max_saturation = max(saturation_degrees.values())
        candidate_nodes = [node for node, sat in saturation_degrees.items() if sat == max_saturation]
        if len(candidate_nodes) > 1:
            return max(candidate_nodes, key=lambda node: self.graph.degree(node))
        return candidate_nodes[0]

    def _get_first_legal_color(self, node, coloring):
        neighbor_colors = {coloring[neighbor] for neighbor in self.adj[node] if neighbor in coloring}
        color = 0
        while True:
            if color not in neighbor_colors: return color
            color += 1
    
    def solve(self):
        print("\n--- [Greedy DSATUR] Запуск жадібного алгоритму ---")
        start_time = time.perf_counter()
        coloring = {}
        for i in range(len(self.nodes)):
            current_node = self._get_dsatur_node(coloring)
            if current_node is None: break
            desc_select = f"[Greedy DSATUR] Крок {i+1}:\nВершина '{current_node}' обрана як найбільш насичена."
            self.history.append({'type': 'node_selection', 'current_node': current_node, 'coloring': dict(coloring), 'description': desc_select})
            chosen_color = self._get_first_legal_color(current_node, coloring)
            coloring[current_node] = chosen_color
            desc_assign = f"[Greedy DSATUR] Крок {i+1}:\nВершині '{current_node}' призначено перший легальний колір: {chosen_color+1}."
            self.history.append({'type': 'color_assignment', 'current_node': current_node, 'coloring': dict(coloring), 'description': desc_assign})
        
        duration = time.perf_counter() - start_time
        num_colors_used = len(set(coloring.values()))
        print(f"+++ [Greedy DSATUR] Завершено. Використано кольорів: {num_colors_used}. Час: {duration:.4f} сек. +++")
        desc_done = f"[Greedy DSATUR] АЛГОРИТМ ЗАВЕРШЕНО.\nКількість кольорів: {num_colors_used}.\nЗагальний час: {duration:.4f} сек."
        self.history.append({'type': 'search_end', 'current_node': None, 'coloring': coloring, 'description': desc_done})
        return coloring, num_colors_used, self.history

class GreedyLDF:
    def __init__(self, graph):
        self.graph = graph
        self.nodes = sorted(list(graph.nodes()), key=lambda n: graph.degree(n), reverse=True)
        self.adj = self.graph.adj
        self.history = []
    
    def _get_first_legal_color(self, node, coloring):
        neighbor_colors = {coloring[neighbor] for neighbor in self.adj[node] if neighbor in coloring}
        color = 0
        while True:
            if color not in neighbor_colors: return color
            color += 1

    def solve(self):
        print("\n--- [Greedy LDF] Запуск жадібного алгоритму (Largest Degree First) ---")
        start_time = time.perf_counter()
        coloring = {}
        sorted_nodes_str = ", ".join(map(str, self.nodes[:5])) + "..."
        desc_initial = f"[Greedy LDF] Вершини відсортовано за ступенем (спаданням):\n{sorted_nodes_str}"
        self.history.append({'type': 'node_selection', 'current_node': None, 'coloring': {}, 'description': desc_initial})

        for i, node in enumerate(self.nodes):
            desc_select = f"[Greedy LDF] Крок {i+1}:\nОбираємо наступну вершину зі списку: '{node}'."
            self.history.append({'type': 'node_selection', 'current_node': node, 'coloring': dict(coloring), 'description': desc_select})
            
            chosen_color = self._get_first_legal_color(node, coloring)
            coloring[node] = chosen_color
            
            desc_assign = f"[Greedy LDF] Крок {i+1}:\nВершині '{node}' призначено перший легальний колір: {chosen_color+1}."
            self.history.append({'type': 'color_assignment', 'current_node': node, 'coloring': dict(coloring), 'description': desc_assign})
        
        duration = time.perf_counter() - start_time
        num_colors_used = len(set(coloring.values()))
        print(f"+++ [Greedy LDF] Завершено. Використано кольорів: {num_colors_used}. Час: {duration:.4f} сек. +++")
        desc_done = f"[Greedy LDF] АЛГОРИТМ ЗАВЕРШЕНО.\nКількість кольорів: {num_colors_used}.\nЗагальний час: {duration:.4f} сек."
        self.history.append({'type': 'search_end', 'current_node': None, 'coloring': coloring, 'description': desc_done})
        return coloring, num_colors_used, self.history

class InteractivePlot:
    def __init__(self, graph, histories, max_k):
        self.graph = graph
        self.histories = histories
        self.modes = list(histories.keys())
        self.current_mode = self.modes[0] if self.modes else None
        
        self.k = max_k
        self.pos = nx.spring_layout(graph, seed=42)

        self.color_palette = plt.get_cmap('viridis', self.k + 5)
        self.node_colors = [mcolors.to_hex(self.color_palette(i)) for i in range(self.k + 5)]
        
        self.current_steps = {mode: 0 for mode in self.modes}
        self.iteration_end_steps = {mode: [i for i, step in enumerate(hist) if step.get('type') in ['global_update', 'solution_found', 'search_end', 'ant_done']] for mode, hist in self.histories.items()}

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title('Візуалізатор алгоритмів розфарбовування графу')

    def _draw_step(self):
        if self.current_mode is None: return
        self.ax.clear()
        
        current_step_index = self.current_steps[self.current_mode]
        mode_history = self.histories[self.current_mode]
        step_data = mode_history[current_step_index]
        
        coloring, current_node, description = step_data['coloring'], step_data['current_node'], step_data['description']
        
        node_color_list = []
        for node in self.graph.nodes():
            if node in coloring and coloring[node] < len(self.node_colors):
                node_color_list.append(self.node_colors[coloring[node]])
            else:
                node_color_list.append('lightgrey')

        edge_colors = ['red' if u in coloring and v in coloring and coloring.get(u) == coloring.get(v) else 'black' for u, v in self.graph.edges()]
        
        nx.draw_networkx_edges(self.graph, self.pos, edge_color=edge_colors, alpha=0.7, ax=self.ax)
        nodes_plot = nx.draw_networkx_nodes(self.graph, self.pos, node_color=node_color_list, node_size=500, ax=self.ax)
        nodes_plot.set_edgecolor('black')
        if current_node:
            node_idx = list(self.graph.nodes()).index(current_node)
            nx.draw_networkx_nodes(self.graph, self.pos, nodelist=[current_node], node_size=700,
                                   node_color=[node_color_list[node_idx]],
                                   linewidths=3.0, edgecolors='gold', ax=self.ax)
        nx.draw_networkx_labels(self.graph, self.pos, font_color='black', font_weight='bold', ax=self.ax)
        
        mode_keys = ", ".join([str(i+1) for i in range(len(self.modes))])
        title = (f"Режим: {self.current_mode.upper()} (Перемкнути: {mode_keys})\n"
                 f"Крок {current_step_index + 1}/{len(mode_history)}\n"
                 f"(← →, Home, End, Ctrl+←, Ctrl+→)")
        self.ax.set_title(title, fontsize=14)
        
        self.fig.texts.clear()
        self.fig.text(0.5, 0.08, description, ha='center', va='top', fontsize=12, wrap=True,
                      bbox=dict(boxstyle='round,pad=0.5', fc='ivory', ec='black', lw=1, alpha=0.9))
        self.fig.tight_layout(rect=[0, 0.15, 1, 0.95])
        self.fig.canvas.draw()

    def _on_key_press(self, event):
        key = event.key
        
        if key.isdigit() and 1 <= int(key) <= len(self.modes):
            new_mode_idx = int(key) - 1
            if self.modes[new_mode_idx] != self.current_mode:
                self.current_mode = self.modes[new_mode_idx]
                print(f"\nПереключено на режим: {self.current_mode}")
                self._draw_step()
            return

        step = self.current_steps[self.current_mode]
        history_len = len(self.histories[self.current_mode])
        
        if key == 'right':
            if step < history_len - 1: self.current_steps[self.current_mode] += 1
        elif key == 'left':
            if step > 0: self.current_steps[self.current_mode] -= 1
        elif key == 'end':
            self.current_steps[self.current_mode] = history_len - 1
        elif key == 'home':
            self.current_steps[self.current_mode] = 0
        elif key == 'ctrl+right':
            end_steps = self.iteration_end_steps[self.current_mode]
            for step_idx in end_steps:
                if step_idx > step: self.current_steps[self.current_mode] = step_idx; break
        elif key == 'ctrl+left':
            end_steps = self.iteration_end_steps[self.current_mode]
            for step_idx in reversed(end_steps):
                if step_idx < step: self.current_steps[self.current_mode] = step_idx; break
        else: return
        self._draw_step()

    def show(self):
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self._draw_step()
        print("\nВІДКРИТО ІНТЕРАКТИВНЕ ВІКНО:")
        print("1. Клацніть на вікно, щоб зробити його активним.")
        print("2. Використовуйте клавіші для навігації та перемикання режимів:")
        for i, mode in enumerate(self.modes):
            print(f"   {i+1}       : Переключитись на візуалізацію {mode}")
        print("   →       : Наступний крок")
        print("   ←       : Попередній крок")
        print("   Home    : Перейти на самий початок")
        print("   End     : Перейти в самий кінець")
        print("   Ctrl+→  : До кінця наступної ітерації (тільки для ACO)")
        print("   Ctrl+←  : До кінця попередньої ітерації (тільки для ACO)")
        plt.show()

def get_default_config():
    print("INFO: Використовується дефолтна конфігурація.")
    return {
      "graph": {
        "edges": [
          [1, 3], [1, 4], [1, 5], [1, 9], [1, 10], [1, 11], [1, 13], [1, 14], [1, 16], [1, 20],
          [2, 4], [2, 7], [2, 10], [2, 12], [2, 13], [2, 15], [2, 18], [2, 19],
          [3, 6], [3, 12], [3, 13], [3, 17], [3, 18], [3, 20],
          [4, 6], [4, 7], [4, 11], [4, 13], [4, 15], [4, 16], [4, 17], [4, 18], [4, 19],
          [5, 6], [5, 7], [5, 10], [5, 11], [5, 15], [5, 16], [5, 17], [5, 18],
          [6, 12], [6, 13], [6, 19],
          [7, 8], [7, 9], [7, 10], [7, 12], [7, 13], [7, 17], [7, 18], [7, 19],
          [8, 10], [8, 11], [8, 12], [8, 13], [8, 14], [8, 15], [8, 19],
          [9, 11], [9, 13], [9, 17], [9, 18], [9, 19],
          [10, 12], [10, 13], [10, 15], [10, 19],
          [11, 13], [11, 18], [11, 20],
          [12, 14], [12, 15], [12, 16], [12, 17], [12, 20],
          [13, 15], [13, 16], [13, 17], [13, 19],
          [14, 15], [14, 16], [14, 19], [14, 20],
          [15, 16], [15, 17], [15, 18],
          [16, 18], [16, 20],
          [17, 18], [17, 19],
          [18, 19],
          [19, 20]
        ]
      },
      "aco_params": {
        "num_ants": 100, "alpha": 2.0, "beta": 4.0, "rho": 0.4, "q": 100
      },
      "solver_params": {
        "start_k_strategy": "clique", "iterations_per_k": 100
      }
    }

def load_config(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
            print(f"INFO: Конфігурацію успішно завантажено з файлу: {filepath}")
            return config
    except FileNotFoundError:
        print(f"ПОМИЛКА: Конфігураційний файл не знайдено за шляхом: {filepath}", file=sys.stderr)
        return None
    except json.JSONDecodeError:
        print(f"ПОМИЛКА: Некоректний формат JSON у файлі: {filepath}", file=sys.stderr)
        return None

def create_graph_from_config(config):
    G = nx.Graph()
    edges = config['graph']['edges']
    G.add_edges_from(edges)
    print(f"Граф успішно створено: {G.number_of_nodes()} вершин, {G.number_of_edges()} ребер.")
    return G


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Розв'язання задачі розфарбовування графу за допомогою ACO та жадібних алгоритмів.",
        epilog="Приклад запуску: python pz1_ACO.py [шлях_до_config.json]"
    )
    parser.add_argument(
        "config_file",
        type=str,
        nargs='?',
        default=None,
        help="Опціональний шлях до конфігураційного файлу JSON."
    )
    args = parser.parse_args()

    config = None
    if args.config_file:
        config = load_config(args.config_file)
    
    if config is None:
        config = get_default_config()

    G = create_graph_from_config(config)
    
    start_k_strategy = config.get('solver_params', {}).get('start_k_strategy', 'clique')
    if isinstance(start_k_strategy, int):
        start_k = start_k_strategy
        print(f"Встановлено початкове k = {start_k} з конфігурації.")
    else:
        print("Розрахунок нижньої межі кількості кольорів (розмір максимальної кліки)...")
        max_clique, w = nx.algorithms.clique.max_weight_clique(G, weight=None)
        start_k = len(max_clique)
        print(f"Нижня межа: {start_k}. Починаємо з k = {start_k}.")

    aco_params = config.get('aco_params', {})
    iterations_per_k = config.get('solver_params', {}).get('iterations_per_k', 100)
    

    solver_aco_dsatur = ChromaticSolver(graph=G, **aco_params, node_selection_strategy='dsatur')
    _, k_aco_dsatur, history_aco_dsatur = solver_aco_dsatur.solve(start_k=start_k, iterations_per_k=iterations_per_k)

    solver_aco_random = ChromaticSolver(graph=G, **aco_params, node_selection_strategy='random')
    _, k_aco_random, history_aco_random = solver_aco_random.solve(start_k=start_k, iterations_per_k=iterations_per_k)
    
    solver_greedy_dsatur = GreedyDSATUR(graph=G)
    _, k_greedy_dsatur, history_greedy_dsatur = solver_greedy_dsatur.solve()

    solver_greedy_ldf = GreedyLDF(graph=G)
    _, k_greedy_ldf, history_greedy_ldf = solver_greedy_ldf.solve()
    
    all_histories = {
        'ACO+DSATUR': history_aco_dsatur,
        'ACO+Random': history_aco_random,
        'Greedy DSATUR': history_greedy_dsatur,
        'Greedy LDF': history_greedy_ldf
    }
    
    max_k_for_palette = max(start_k, k_greedy_dsatur, k_greedy_ldf, 1)

    if any(all_histories.values()):
        plot = InteractivePlot(G, all_histories, max_k=max_k_for_palette)
        plot.show()