import math
import time
import random
import multiprocessing
import psutil # type: ignore
import os
import copy
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from collections import deque, defaultdict
from scipy import stats # type: ignore

# Constants Configuration
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.01

SOLUTION_POOL_MAX_SIZE = 10

GDA_PARAMS = {
    'CHAOS_PARAM': 3.9,
    'INITIAL_BOUNDARY_MULTIPLIER': 1.25,
    'RAIN_SPEED_BOOK_FACTOR': 0.1,
    'CHAOTIC_PERTURBATION_THRESHOLDS': (0.3, 0.6),  # Lower/Upper bounds
    'LAHC_THRESHOLD_MULTIPLIER': 0.98,
    'BOUNDARY_RESET_THRESHOLD': 0.01,
    'BOUNDARY_RESET_MULTIPLIER': 1.5,
    'STAGNATION_LIMIT': 100,
    'POOL_SYNC_FREQUENCY': 50
}

PARALLEL_CONFIG = {
    'BASE_PHASE_RATIO': 0.05,
    'MIN_PHASE_TIME': 15,  # seconds
    'MAX_PHASE_RATIO': 0.25,
    'MIN_PHASES': 5,
    'PHASE_EXTENSION_FACTOR': 1.2,
    'PHASE_REDUCTION_FACTOR': 0.8,
    'PERFORMANCE_FACTORS': (1.1, 0.9),  # Improvement/Stagnation
    'PROGRESS_THRESHOLDS': (0.2, 0.6),  # Exploration/Main/Exploitation
    'TIME_FACTORS': (0.8, 1.0, 1.2),    # Corresponding to progress thresholds
    'ABSOLUTE_MIN_PHASE_TIME': 5
}


class EarlyStopping:
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.counter = 0

    def should_stop(self, current_score):
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

class SharedMemoryManager:
    def __init__(self):
        self.strategy_weights = defaultdict(lambda: 1.0)
        self.lock = multiprocessing.Lock()

    def update_weights(self, strategy, improvement):
        with self.lock:
            self.strategy_weights[strategy] *= 1.1 if improvement > 0 else 0.9

class SolutionPool:
    def __init__(self, max_size=SOLUTION_POOL_MAX_SIZE):
        self.pool = []
        self.lock = multiprocessing.Lock()
        self.max_size = max_size

    def add_solution(self, solution):
        with self.lock:
            if len(self.pool) < self.max_size:
                self.pool.append(copy.deepcopy(solution))
            else:
                self.pool.sort(key=lambda x: x.fitness_score)
                if solution.fitness_score > self.pool[0].fitness_score:
                    self.pool[0] = copy.deepcopy(solution)

    def get_best(self):
        with self.lock:
            return max(self.pool, key=lambda x: x.fitness_score) if self.pool else None

    def similarity(self, sol1, sol2):
        set1 = set(sol1.scanned_books)
        set2 = set(sol2.scanned_books)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

class EnhancedGreatDeluge:
    def __init__(self, solver, data, strategy_params, initial_solution=None):
        self.solver = solver
        self.data = data
        self.strategy_params = strategy_params
        self.chaos_param = strategy_params.get('chaos_param', GDA_PARAMS['CHAOS_PARAM'])
        self.shared_memory = SharedMemoryManager()
        self.solution_pool = SolutionPool()
        self.iteration = 0
        
        self.current_solution = initial_solution or self.solver.generate_initial_solution_grasp(data)
        self.current_score = self.current_solution.fitness_score
        self.best_solution = copy.deepcopy(self.current_solution)
        self.best_score = self.current_score
        
        self.B0 = self.best_score * GDA_PARAMS['INITIAL_BOUNDARY_MULTIPLIER']
        self.B = self.B0
        self.delta_B = self.calculate_initial_rain_speed()
        self.metrics = {
            'diversity': 0.0,
            'acceptance_rate': 0.0,
            'strategy_effectiveness': defaultdict(float),
            'boundary_resets': 0
        }

    def calculate_initial_rain_speed(self):
        num_books = len(self.data.scores)
        num_libs = len(self.data.libs)
        return (num_books * GDA_PARAMS['RAIN_SPEED_BOOK_FACTOR']) / (num_libs * math.log(num_libs + 1))

    def chaotic_perturbation(self, solution):
        x = random.random()
        for _ in range(5):
            x = self.chaos_param * x * (1 - x)
        
        if x < GDA_PARAMS['CHAOTIC_PERTURBATION_THRESHOLDS'][0]:
            return self.solver.tweak_solution_swap_signed(solution, self.data)
        elif x < GDA_PARAMS['CHAOTIC_PERTURBATION_THRESHOLDS'][1]:
            return self.solver.tweak_solution_swap_last_book(solution, self.data)
        else:
            return self.solver.hill_climbing_combined(self.data, iterations=10)[1]

    def hybrid_acceptance(self, candidate_score):
        lahc_threshold = self.best_score * GDA_PARAMS['LAHC_THRESHOLD_MULTIPLIER']
        return candidate_score >= self.B or candidate_score > lahc_threshold

    def update_strategy_weights(self, improvement):
        strategy = self.strategy_params['type']
        self.shared_memory.update_weights(
            strategy, 
            improvement / self.best_score if self.best_score > 0 else 0
        )

    def dynamic_boundary_adjustment(self):
        base_B = self._adjust_boundary()
        chaos_factor = 1 + 0.1 * math.sin(self.iteration * math.pi / 50)
        return base_B * chaos_factor

    def _adjust_boundary(self):
        strategy = self.strategy_params['type']
        params = self.strategy_params['params']
        i = self.iteration
        
        boundary_strategies = {
            'linear': lambda: self.B0 - params['r'] * i,
            'multiplicative': lambda: self.B0 * (params['alpha'] ** i),
            'stepwise': lambda: self.B0 - (i // params['step_size']) * params['d'],
            'logarithmic': lambda: self.B0 - params['r'] * math.log(i + 1),
            'quadratic': lambda: self.B0 - params['r'] * (i ** 2)
        }
        
        return boundary_strategies[strategy]()

    def _calculate_pool_diversity(self):
        if len(self.solution_pool.pool) < 2:
            return 0.0
        
        books_scanned = [len(solution.scanned_books) for solution in self.solution_pool.pool]
        mean = np.mean(books_scanned)
        return np.std(books_scanned) / mean if mean != 0 else 0.0

    def run(self, max_time=300):
        start_time = time.time()
        self.consecutive_stagnation = 0
        accepted = 0
        total = 0
        min_boundary = self.best_score * GDA_PARAMS['BOUNDARY_RESET_THRESHOLD']
        
        while not self.should_terminate(start_time, max_time):
            candidate = self.chaotic_perturbation(self.current_solution)
            total += 1
            
            if self.iteration % GDA_PARAMS['POOL_SYNC_FREQUENCY'] == 0:
                pool_solution = self.solution_pool.get_best()
                if pool_solution and pool_solution.fitness_score > self.current_score:
                    self.current_solution = copy.deepcopy(pool_solution)
                    self.current_score = self.current_solution.fitness_score

            self.B = self.dynamic_boundary_adjustment()
            
            if self.B < min_boundary:
                self.B = self.best_score * GDA_PARAMS['BOUNDARY_RESET_MULTIPLIER']
                self.metrics['boundary_resets'] += 1

            if self.hybrid_acceptance(candidate.fitness_score):
                accepted += 1
                improvement = candidate.fitness_score - self.current_score
                self.current_solution = candidate
                self.current_score = candidate.fitness_score
                
                if self.current_score > self.best_score:
                    self.best_score = self.current_score
                    self.best_solution = copy.deepcopy(self.current_solution)
                    self.consecutive_stagnation = 0
                    self.solution_pool.add_solution(self.best_solution)
                else:
                    self.consecutive_stagnation += 1
                
                self.update_strategy_weights(improvement)
                self.delta_B *= 0.99 if improvement > 0 else 1.01

            self.metrics.update({
                'diversity': self._calculate_pool_diversity(),
                'acceptance_rate': accepted / total if total > 0 else 0.0,
                'strategy_effectiveness': copy.copy(self.shared_memory.strategy_weights)
            })

            self.iteration += 1

        return self.best_score, self.best_solution

    def should_terminate(self, start_time, max_time):
        time_exceeded = (time.time() - start_time) >= max_time
        stagnation = self.consecutive_stagnation >= GDA_PARAMS['STAGNATION_LIMIT']
        min_boundary = self.best_score * GDA_PARAMS['BOUNDARY_RESET_THRESHOLD']
        return any([time_exceeded, stagnation, self.B < min_boundary])

class ParallelGDARunner:
    def __init__(self, solver, data):
        self.solver = solver
        self.data = data
        self.phase_history = []
        self.early_stop = EarlyStopping(patience=5, min_delta=0.01)
        self.max_total_time = 300
        self.start_time = time.time()
        
        # Dynamic phase configuration
        self.base_phase_ratio = PARALLEL_CONFIG['BASE_PHASE_RATIO']  # 5% of total time
        self.min_phase_time = PARALLEL_CONFIG['MIN_PHASE_TIME']      # 15 seconds minimum
        self.max_phase_ratio = PARALLEL_CONFIG['MAX_PHASE_RATIO']    # 25% max phase duration
        self.min_phases = PARALLEL_CONFIG['ABSOLUTE_MIN_PHASE_TIME'] # Minimum phases to execute
        
        # Core allocation
        total_cores = multiprocessing.cpu_count()
        cores_per_strategy = max(1, total_cores // 5)
        self.cpu_assignments = {}
        
        for i in range(5):
            start_core = i * cores_per_strategy
            end_core = min((i + 1) * cores_per_strategy, total_cores)
            cores = list(range(start_core, end_core))
            
            self.cpu_assignments[i] = {
                'cores': cores,
                'strategy': ['linear', 'multiplicative', 'stepwise', 
                            'logarithmic', 'quadratic'][i]
            }

    def set_cpu_affinity(self, cores):
        proc = psutil.Process(os.getpid())
        try:
            proc.cpu_affinity(cores)
        except Exception as e:
            print(f"CPU affinity error: {str(e)}")

    def strategy_worker(self, strategy, cpu_cores, phase_time, initial_solution):
        try:
            self.set_cpu_affinity(cpu_cores)
            optimizer = EnhancedGreatDeluge(
                self.solver, 
                self.data, 
                strategy,
                initial_solution=copy.deepcopy(initial_solution)
            )
            score, solution = optimizer.run(max_time=phase_time)
            return (score, solution)
        except Exception as e:
            print(f"Strategy error: {str(e)}")
            return (0, None)

    def calculate_base_phase_time(self, max_total_time):
        base_time = max_total_time * self.base_phase_ratio
        return max(
            self.min_phase_time,
            min(base_time, max_total_time * self.max_phase_ratio)
        )

    def adaptive_phase_time(self, phase, remaining_time, max_total_time):
        # Calculate progress through total runtime
        elapsed = max_total_time - remaining_time
        progress = elapsed / max_total_time if max_total_time > 0 else 0
        
        # Base phase time calculation
        base_time = self.calculate_base_phase_time(max_total_time)
        
        # Progress-based adjustment
        if progress < PARALLEL_CONFIG['PROGRESS_THRESHOLDS'][0]:
            time_factor = PARALLEL_CONFIG['TIME_FACTORS'][0]  # Faster phases early
        elif progress < PARALLEL_CONFIG['PROGRESS_THRESHOLDS'][1]:
            time_factor = PARALLEL_CONFIG['TIME_FACTORS'][1]  # Normal pace
        else:
            time_factor = PARALLEL_CONFIG['TIME_FACTORS'][2]  # Slower phases late

        # Performance-based adjustment
        if len(self.phase_history) > 1:
            last_improvement = self.phase_history[-1][1] - self.phase_history[-2][1]
            if last_improvement > 0:
                time_factor *= PARALLEL_CONFIG['PERFORMANCE_FACTORS'][0]
            else:
                time_factor *= PARALLEL_CONFIG['PERFORMANCE_FACTORS'][1]

        # Calculate final phase time with constraints
        phase_time = base_time * time_factor
        return min(
            max(phase_time, self.min_phase_time),
            remaining_time * 0.9,  # Leave 10% buffer
            max_total_time * self.max_phase_ratio
        )

    def run_iterative_phases(self, max_total_time=300):
        self.max_total_time = max_total_time
        self.start_time = time.time()
        best_score = 0
        current_best = self.solver.generate_initial_solution_grasp(self.data)
        phase = 0
        
        while (remaining_time := max_total_time - (time.time() - self.start_time)) > 0:
            # Calculate adaptive phase time
            phase_time = self.adaptive_phase_time(phase, remaining_time, max_total_time)
            
            # Enforce minimum phases
            min_phases = max(self.min_phases, int(max_total_time / 120))
            if phase < min_phases:
                phase_time = min(phase_time, remaining_time / (min_phases - phase))
                
            # Ensure we don't exceed remaining time
            phase_time = min(phase_time, remaining_time)
            if phase_time < 5:  # Absolute minimum
                break

            print(f"\n=== PHASE {phase} ===")
            print(f"Phase time: {phase_time:.1f}s | Remaining: {remaining_time:.1f}s")

            with multiprocessing.Pool(processes=5) as pool:
                args = [
                    (
                        {'type': self.cpu_assignments[i]['strategy'], 
                         'params': self.get_strategy_params(i)},
                        self.cpu_assignments[i]['cores'],
                        phase_time,
                        current_best
                    )
                    for i in range(5)
                ]
                
                results = pool.starmap(self.strategy_worker, args)

            valid_results = [r for r in results if r[1] is not None]
            if valid_results:
                phase_best_score, phase_best_solution = max(valid_results, key=lambda x: x[0])
                
                if phase_best_score > best_score:
                    best_score = phase_best_score
                    current_best = copy.deepcopy(phase_best_solution)
                    print(f"New global best: {best_score}")

            self.phase_history.append((
                phase,
                best_score,
                time.time() - self.start_time,
                current_best.metrics['diversity'] if hasattr(current_best, 'metrics') else 0.0
            ))

            if self.early_stop.should_stop(best_score):
                print(f"Early stopping at phase {phase}")
                break

            phase += 1

        return best_score, current_best

    def get_strategy_params(self, strategy_idx):
        params_map = {
            0: {'r': 0.1},
            1: {'alpha': 0.99},
            2: {'step_size': 10, 'd': 50},
            3: {'r': 2},
            4: {'r': 0.01}
        }
        return params_map[strategy_idx]

    def analyze_performance(self):
        phases, scores, times, _ = zip(*self.phase_history)
        
        plt.figure(figsize=(12,6))
        plt.plot(times, scores, 'b-', marker='o')
        plt.xlabel('Time (s)')
        plt.ylabel('Best Score')
        plt.title('Score Improvement Over Time')
        plt.show()
        
        plt.figure(figsize=(12,6))
        plt.bar(phases, scores)
        plt.xlabel('Phase Number')
        plt.ylabel('Best Score')
        plt.title('Per-Phase Performance')
        plt.show()
        
        improvements = [scores[i+1]-scores[i] for i in range(len(scores)-1)]
        plt.figure(figsize=(12,6))
        plt.plot(improvements)
        plt.xlabel('Phase Transition')
        plt.ylabel('Score Delta')
        plt.title('Marginal Improvements Between Phases')
        plt.show()

    def improvement_potential(self, confidence=0.95):
        if len(self.phase_history) < 2:
            return float('inf')
        
        scores = [s for _,s,_,_ in self.phase_history]
        improvements = np.diff(scores)
        
        mean = np.mean(improvements)
        sem = stats.sem(improvements)
        ci = stats.t.interval(confidence, len(improvements)-1, loc=mean, scale=sem)
        
        time_remaining = max(0, self.max_total_time - (time.time() - self.start_time))
        return ci[1] * (time_remaining / self.calculate_base_phase_time(self.max_total_time))

    def should_continue(self, current_score, min_improvement=1000):
        return self.improvement_potential() > min_improvement