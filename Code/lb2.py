import numpy as np
import matplotlib.pyplot as plt
import time
import json
import argparse
import sys
from matplotlib.animation import FuncAnimation

class LevyFunction:
    def __init__(self, dim=10):
        self.dim = dim
        self.bounds = np.array([[-10, 10]] * dim)
        # Глобальний мінімум f(x) = 0 знаходиться в точці x_i = 1 для всіх i
        self.global_minimum_pos = np.ones(dim)
        self.global_minimum_val = 0.0

    def evaluate(self, x):
        x = np.asarray(x)
        d = self.dim

        # wi = 1 + (xi - 1) / 4
        w = 1 + (x - 1) / 4

        term1 = (np.sin(np.pi * w[0]))**2
        
        term3 = ((w[d-1] - 1)**2) * (1 + (np.sin(2 * np.pi * w[d-1]))**2)
        
        wi_minus_1 = w[:-1]
        sum_term = np.sum(((wi_minus_1 - 1)**2) * (1 + 10 * (np.sin(np.pi * wi_minus_1 + 1))**2))
        
        return term1 + sum_term + term3

class ParticleSwarmOptimization:
    def __init__(self, objective_func, bounds, num_particles, dim, w, c1, c2, boundary_strategy='clamping'):
        self.objective_func = objective_func
        self.bounds = bounds
        self.low_b, self.high_b = self.bounds[:, 0], self.bounds[:, 1]
        self.num_particles = num_particles
        self.dim = dim
        self.w = w  # Коефіцієнт інерції
        self.c1 = c1  # Когнітивний коефіцієнт
        self.c2 = c2  # Соціальний коефіцієнт
        self.boundary_strategy = boundary_strategy
        
        # Ініціалізація стану рою
        self.positions = np.random.uniform(self.low_b, self.high_b, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-abs(self.high_b - self.low_b)*0.1, abs(self.high_b - self.low_b)*0.1, (self.num_particles, self.dim))
        
        self.pbest_pos = self.positions.copy()
        self.pbest_val = np.array([self.objective_func(p) for p in self.positions])
        
        self.gbest_idx = np.argmin(self.pbest_val)
        self.gbest_pos = self.pbest_pos[self.gbest_idx].copy()
        self.gbest_val = self.pbest_val[self.gbest_idx]
        
        self.history = [] # Для візуалізації
        self.gbest_history = []


    def _apply_boundary(self, i):
        pos = self.positions[i]
        vel = self.velocities[i]
        
        if self.boundary_strategy == 'clamping':
            # Стратегія 1: Відсікання (Clamping)
            self.positions[i] = np.clip(pos, self.low_b, self.high_b)
            # Опційно: обнулити швидкість, якщо частинка "вдарилась" об стіну
            hit_low = pos < self.low_b
            hit_high = pos > self.high_b
            self.velocities[i][hit_low | hit_high] = 0
            return "Позиція відсічена до меж."

        elif self.boundary_strategy == 'reflecting':
            # Стратегія 2: Відбиття (Reflection)
            mask_low = pos < self.low_b
            mask_high = pos > self.high_b
            
            if np.any(mask_low) or np.any(mask_high):
                self.positions[i][mask_low] = self.low_b[mask_low] + (self.low_b[mask_low] - pos[mask_low])
                self.positions[i][mask_high] = self.high_b[mask_high] - (pos[mask_high] - self.high_b[mask_high])
                self.velocities[i][mask_low | mask_high] *= -0.5 # Зменшуємо швидкість при відбитті
                return "Відбиття від межі."
            return "Частинка в межах."

        elif self.boundary_strategy == 'wrapping':
            # Стратегія 3: Замикання (Toroidal/Wrapping)
            mask_low = pos < self.low_b
            mask_high = pos > self.high_b
            
            if np.any(mask_low) or np.any(mask_high):
                self.positions[i][mask_low] = self.high_b[mask_low] - (self.low_b[mask_low] - pos[mask_low]) % (self.high_b[mask_low] - self.low_b[mask_low])
                self.positions[i][mask_high] = self.low_b[mask_high] + (pos[mask_high] - self.high_b[mask_high]) % (self.high_b[mask_high] - self.low_b[mask_high])
                return "Замикання простору (телепортація)."
            return "Частинка в межах."
        
        else:
            raise ValueError("Невідома стратегія обробки меж.")


    def _run_one_iteration(self, iteration_id):
        self.history.append({'type': 'iter_start', 'iteration': iteration_id, 'gbest_val': self.gbest_val, 'description': f"Ітерація {iteration_id+1}: Початок. Найкраще значення: {self.gbest_val:.4f}"})

        for i in range(self.num_particles):
            # 1. Оновлення швидкості
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive_vel = self.c1 * r1 * (self.pbest_pos[i] - self.positions[i])
            social_vel = self.c2 * r2 * (self.gbest_pos - self.positions[i])
            self.velocities[i] = self.w * self.velocities[i] + cognitive_vel + social_vel
            self.history.append({'type': 'velocity_update', 'particle_id': i, 'iteration': iteration_id, 'positions': self.positions.copy(), 'velocities': self.velocities.copy(), 'pbest_pos': self.pbest_pos.copy(), 'gbest_pos': self.gbest_pos.copy(), 'description': f"Ітерація {iteration_id+1}, Частинка {i+1}:\nОновлення швидкості."})

            # 2. Оновлення позиції
            self.positions[i] += self.velocities[i]
            
            # 3. Обробка виходу за межі
            boundary_desc = self._apply_boundary(i)
            self.history.append({'type': 'position_update', 'particle_id': i, 'iteration': iteration_id, 'positions': self.positions.copy(), 'velocities': self.velocities.copy(), 'pbest_pos': self.pbest_pos.copy(), 'gbest_pos': self.gbest_pos.copy(), 'description': f"Ітерація {iteration_id+1}, Частинка {i+1}:\nОновлення позиції. {boundary_desc}"})

            # 4. Оцінка нової позиції
            current_val = self.objective_func(self.positions[i])
            
            # 5. Оновлення pbest
            if current_val < self.pbest_val[i]:
                update_desc = f"Оновлено pbest: {self.pbest_val[i]:.4f} -> {current_val:.4f}"
                self.pbest_val[i] = current_val
                self.pbest_pos[i] = self.positions[i].copy()
                
                # 6. Оновлення gbest
                if current_val < self.gbest_val:
                    gbest_old = self.gbest_val
                    self.gbest_val = current_val
                    self.gbest_pos = self.positions[i].copy()
                    update_desc += f"\n!!! НОВИЙ ГЛОБАЛЬНИЙ МІНІМУМ: {gbest_old:.4f} -> {self.gbest_val:.4f} !!!"
                
                self.history.append({'type': 'pbest_update', 'particle_id': i, 'iteration': iteration_id, 'positions': self.positions.copy(), 'velocities': self.velocities.copy(), 'pbest_pos': self.pbest_pos.copy(), 'gbest_pos': self.gbest_pos.copy(), 'description': f"Ітерація {iteration_id+1}, Частинка {i+1}:\n{update_desc}"})
        
        self.gbest_history.append(self.gbest_val)
        self.history.append({'type': 'iter_end', 'iteration': iteration_id, 'gbest_val': self.gbest_val, 'description': f"Ітерація {iteration_id+1}: Кінець. Найкраще значення: {self.gbest_val:.4f}"})

    def run(self, max_iterations=100):
        for i in range(max_iterations):
            self._run_one_iteration(i)
        
        desc_final = f"РОБОТУ ЗАВЕРШЕНО.\nСтратегія: {self.boundary_strategy.upper()}\nНайкраще знайдене значення: {self.gbest_val:.6f}"
        self.history.append({'type': 'final', 'iteration': max_iterations, 'description': desc_final})
        return self.gbest_pos, self.gbest_val, self.history

class PSOSolver:
    def __init__(self, problem, pso_params, boundary_strategy):
        self.problem = problem
        self.pso_params = pso_params
        self.boundary_strategy = boundary_strategy
        self.strategy_name = f"PSO + {boundary_strategy.upper()}"

    def solve(self, max_iterations):
        print(f"\n--- [{self.strategy_name}] Запуск оптимізації ---")
        start_time = time.perf_counter()
        
        pso = ParticleSwarmOptimization(
            objective_func=self.problem.evaluate,
            bounds=self.problem.bounds,
            dim=self.problem.dim,
            boundary_strategy=self.boundary_strategy,
            **self.pso_params
        )
        
        gbest_pos, gbest_val, history = pso.run(max_iterations=max_iterations)
        
        duration = time.perf_counter() - start_time
        print(f"+++ [{self.strategy_name}] Завершено за {duration:.2f} сек. +++")
        print(f"    Найкраще значення: {gbest_val:.6f}")
        print(f"    У точці (перші 4 коорд.): {gbest_pos[:4]}")
        
        return gbest_pos, gbest_val, history

class InteractivePSOPlot:
    def __init__(self, problem, histories):
        self.problem = problem
        self.histories = histories
        self.modes = list(histories.keys())
        self.current_mode = self.modes[0] if self.modes else None
        
        # Налаштування для 2D-візуалізації
        self.fig = plt.figure(figsize=(14, 8))
        self.gs = self.fig.add_gridspec(2, 2)
        self.ax_main = self.fig.add_subplot(self.gs[:, 0])
        self.ax_conv = self.fig.add_subplot(self.gs[0, 1])
        self.ax_text = self.fig.add_subplot(self.gs[1, 1])
        self.ax_text.axis('off')

        self.fig.canvas.manager.set_window_title('Візуалізатор алгоритму PSO для функції Леві')
        
        self._prepare_contour_plot()
        
        self.current_steps = {mode: 0 for mode in self.modes}
        self.iteration_end_steps = {mode: [i for i, step in enumerate(hist) if step.get('type') == 'iter_end'] for mode, hist in self.histories.items()}

    def _prepare_contour_plot(self):
        x = np.linspace(self.problem.bounds[0, 0], self.problem.bounds[0, 1], 150)
        y = np.linspace(self.problem.bounds[1, 0], self.problem.bounds[1, 1], 150)
        X, Y = np.meshgrid(x, y)
        
        # Створюємо 10D-вектори, де перші 2 координати з сітки, а решта - оптимальні (1.0)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.ones(self.problem.dim)
                point[0] = X[i, j]
                point[1] = Y[i, j]
                Z[i, j] = self.problem.evaluate(point)
        
        self.ax_main.contourf(X, Y, Z, levels=np.logspace(0, 3, 20), cmap='viridis_r', alpha=0.7)
        self.ax_main.contour(X, Y, Z, levels=np.logspace(0, 3, 20), colors='gray', linewidths=0.5)
        # Позначаємо глобальний мінімум
        self.ax_main.plot(self.problem.global_minimum_pos[0], self.problem.global_minimum_pos[1], 'r*', markersize=15, label='Глобальний мінімум')
        self.ax_main.set_xlabel('x1')
        self.ax_main.set_ylabel('x2')
        self.ax_main.set_xlim(self.problem.bounds[0])
        self.ax_main.set_ylim(self.problem.bounds[1])
        self.ax_main.legend()

    def _draw_step(self):
        if self.current_mode is None: return

        # Очищення попередніх елементів, що рухаються
        for artist in self.ax_main.lines + self.ax_main.collections:
            if hasattr(artist, 'get_label') and artist.get_label().startswith('_particle_'):
                artist.remove()
        
        current_step_index = self.current_steps[self.current_mode]
        mode_history = self.histories[self.current_mode]
        step_data = mode_history[current_step_index]
        
        positions = step_data.get('positions')
        pbest_pos = step_data.get('pbest_pos')
        gbest_pos = step_data.get('gbest_pos')
        description = step_data.get('description', '')
        
        # Відображення частинок
        if positions is not None:
            self.ax_main.scatter(positions[:, 0], positions[:, 1], c='blue', alpha=0.6, s=30, label='_particle_swarm')
            if pbest_pos is not None:
                 self.ax_main.scatter(pbest_pos[:, 0], pbest_pos[:, 1], c='green', marker='+', s=40, alpha=0.8, label='_particle_pbest')
            if gbest_pos is not None:
                self.ax_main.scatter(gbest_pos[0], gbest_pos[1], c='gold', marker='*', s=150, edgecolors='black', label='_particle_gbest')
        
        # Оновлення заголовку
        mode_keys = ", ".join([str(i+1) for i in range(len(self.modes))])
        title = (f"Режим: {self.current_mode} (Перемкнути: {mode_keys})\n"
                 f"Крок {current_step_index + 1}/{len(mode_history)}\n"
                 f"(← →, Home, End, Ctrl+→/←)")
        self.fig.suptitle(title, fontsize=14)

        # Оновлення текстового опису
        self.ax_text.clear()
        self.ax_text.axis('off')
        self.ax_text.text(0, 0.95, description, ha='left', va='top', fontsize=12, wrap=True,
                          bbox=dict(boxstyle='round,pad=0.5', fc='ivory', ec='black', lw=1, alpha=0.9))

        # Оновлення графіка збіжності
        self.ax_conv.clear()
        gbest_history = [s['gbest_val'] for s in mode_history[:current_step_index+1] if 'gbest_val' in s]
        if gbest_history:
            self.ax_conv.plot(gbest_history, marker='.')
        self.ax_conv.set_title('Збіжність (найкраще значення)')
        self.ax_conv.set_xlabel('Ітерація')
        self.ax_conv.set_ylabel('Значення f(x)')
        self.ax_conv.set_yscale('log')
        self.ax_conv.grid(True)
        
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
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
            next_end_step = next((s for s in end_steps if s > step), history_len - 1)
            self.current_steps[self.current_mode] = next_end_step
        elif key == 'ctrl+left':
            end_steps = self.iteration_end_steps[self.current_mode]
            prev_end_step = next((s for s in reversed(end_steps) if s < step), 0)
            self.current_steps[self.current_mode] = prev_end_step
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
        print("   Ctrl+→  : До кінця наступної ітерації")
        print("   Ctrl+←  : До кінця попередньої ітерації")
        plt.show()

def get_default_config():
    print("INFO: Використовується дефолтна конфігурація.")
    return {
      "problem": {
        "name": "Levy",
        "dim": 10
      },
      "pso_params": {
        "num_particles": 40,
        "w": 0.729,  # Інерція
        "c1": 1.494, # Когнітивний коефіцієнт
        "c2": 1.494  # Соціальний коефіцієнт
      },
      "solver_params": {
        "max_iterations": 50,
        "strategies": ["clamping", "reflecting", "wrapping"]
      }
    }

def load_config(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
            print(f"INFO: Конфігурацію успішно завантажено з файлу: {filepath}")
            return config
    except FileNotFoundError:
        print(f"ПОМИЛКА: Конфігураційний файл не знайдено: {filepath}", file=sys.stderr)
        return None
    except json.JSONDecodeError:
        print(f"ПОМИЛКА: Некоректний формат JSON у файлі: {filepath}", file=sys.stderr)
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Розв'язання задачі мінімізації функції Леві за допомогою PSO.",
        epilog="""Приклади запуску:
  python lb2_PSO_Levy.py                  (запуск з інтерактивною візуалізацією)
  python lb2_PSO_Levy.py --stats         (збір статистики за 10 прогонів)
  python lb2_PSO_Levy.py --stats 50      (збір статистики за 50 прогонів)
  python lb2_PSO_Levy.py my_config.json  (візуалізація з файлу конфігурації)""",
        formatter_class=argparse.RawTextHelpFormatter # Для коректного відображення прикладів
    )
    # Аргумент для файлу конфігурації (позиційний, опціональний)
    parser.add_argument(
        "config_file", type=str, nargs='?', default=None,
        help="Опціональний шлях до конфігураційного файлу JSON."
    )
    # Аргумент-прапорець для режиму статистики (опціональний)
    parser.add_argument(
        '--stats',
        type=int,
        nargs='?',
        const=10,  # Значення за замовчуванням, якщо прапорець вказано без числа
        default=None, # Значення, якщо прапорець взагалі відсутній
        help="Запустити в режимі збору статистики. Опціонально можна вказати кількість прогонів (за замовчуванням: 10)."
    )
    args = parser.parse_args()

    # Завантаження конфігурації
    config = load_config(args.config_file) if args.config_file else get_default_config()
    if not config:
        sys.exit(1)

    # Створення екземпляру задачі
    problem = LevyFunction(dim=config['problem']['dim'])
    
    # РЕЖИМ 1: ЗБІР СТАТИСТИКИ (якщо є прапорець --stats)
    if args.stats is not None:
        NUM_RUNS = args.stats
        print(f"Запуск у режимі збору статистики ({NUM_RUNS} прогонів)...")
        
        strategies_to_run = config['solver_params']['strategies']
        max_iterations = config['solver_params']['max_iterations']

        all_results = {strategy: [] for strategy in strategies_to_run}
        all_times = {strategy: [] for strategy in strategies_to_run}

        for i in range(NUM_RUNS):
            print(f"\n--- Прогін {i + 1}/{NUM_RUNS} ---")
            for strategy in strategies_to_run:
                start_time = time.perf_counter()
                solver = PSOSolver(problem, config['pso_params'], boundary_strategy=strategy)
                _, val, _ = solver.solve(max_iterations)
                duration = time.perf_counter() - start_time
                
                all_results[strategy].append(val)
                all_times[strategy].append(duration)
        
        print("\n\n" + "="*40)
        print("====== СТАТИСТИЧНІ РЕЗУЛЬТАТИ ======")
        print("="*40)

        for strategy in strategies_to_run:
            results_np = np.array(all_results[strategy])
            times_np = np.array(all_times[strategy])
            
            best_val = np.min(results_np)
            mean_val = np.mean(results_np)
            std_val = np.std(results_np)
            mean_time = np.mean(times_np)
            
            print(f"\nСтратегія: {strategy.upper()}")
            print(f"  Найкращий результат (Best):       {best_val:.6f}")
            print(f"  Середній результат (Mean):      {mean_val:.6f}")
            print(f"  Стандартне відхилення (Std):    {std_val:.6f}")
            print(f"  Середній час виконання:         {mean_time:.4f} сек")
            print(f"  Повний результат (Best/Mean±Std): {best_val:.6f} / {mean_val:.6f} ± {std_val:.6f}")

    # РЕЖИМ 2: ІНТЕРАКТИВНА ВІЗУАЛІЗАЦІЯ (якщо прапорця --stats немає)
    else:
        print("Запуск у режимі інтерактивної візуалізації...")
        
        all_histories = {}
        final_results = {}
        
        strategies_to_run = config['solver_params']['strategies']
        max_iterations = config['solver_params']['max_iterations']
        
        for strategy in strategies_to_run:
            solver = PSOSolver(problem, config['pso_params'], boundary_strategy=strategy)
            pos, val, history = solver.solve(max_iterations)
            all_histories[f"PSO-{strategy.upper()}"] = history
            final_results[strategy] = val
        
        print("\n\n=== ПІДСУМКОВІ РЕЗУЛЬТАТИ (одиничний запуск) ===")
        for strategy, value in final_results.items():
            print(f"Стратегія '{strategy.upper()}':\tНайкраще значення = {value:.6f}")
        print(f"Теоретичний мінімум: {problem.global_minimum_val}")
        
        if any(all_histories.values()):
            plot = InteractivePSOPlot(problem, all_histories)
            plot.show()