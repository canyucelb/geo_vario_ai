import os
import sys
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

class SISIMWorkflow:
    """
    Encapsulates SISIM and PostSim workflows for splitting data, updating 
    parameter files, running simulations, and reading outputs.
    """

    def __init__(self, sisim_param: str = 'sisim.par', postsim_param: str = 'postsim.par'):
        self.sisim_param = sisim_param
        self.postsim_param = postsim_param

    def split_data_file(
        self,
        source_data: str,
        sample_file: str = 'sample_data.txt',
        true_file: str = 'true_data.txt',
        frac: float = 0.7
    ) -> None:
        with open(source_data, 'r') as f:
            lines = f.readlines()
        header = lines[:7]
        body = lines[7:]
        data = np.loadtxt(body)
        primary_vals = data[:, 2]
        unique_vals = np.unique(primary_vals)
        n_train = int(frac * len(unique_vals))
        chosen_vals = np.random.choice(unique_vals, size=n_train, replace=False)
        train_data = data[np.isin(primary_vals, chosen_vals)]
        test_data = data[~np.isin(primary_vals, chosen_vals)]

        np.savetxt(sample_file, train_data, header=''.join(header), comments='', delimiter=' ', fmt='%.5f')
        np.savetxt(true_file, test_data, header=''.join(header), comments='', delimiter=' ', fmt='%.5f')

    def modify_sisim_datafile(self, datafile: str) -> None:
        with open(self.sisim_param, 'r') as f:
            lines = f.readlines()
        lines[8]  = f"{datafile}                   -file with data\n"
        lines[19] = f"{datafile}                   -   file with tabulated values\n"
        with open(self.sisim_param, 'w') as f:
            f.writelines(lines)

    def update_sisim_variogram(
        self,
        *,
        nst: int,
        nugget: float,
        it: np.ndarray,
        cc: np.ndarray,
        ang1: np.ndarray,
        ang2: np.ndarray,
        ang3: np.ndarray,
        a_hmax: np.ndarray,
        a_hmin: np.ndarray,
        a_hvert: np.ndarray
    ) -> None:
        with open(self.sisim_param, 'r') as f:
            lines = f.readlines()
        lines[5]  = "1                             -number thresholds/categories\n"
        lines[9]  = "1   2   0   3                 -   columns for X,Y,Z, variable\n"
        lines[11] = "1   2   0   3 4 5 6 7         -   columns for variable, weight\n"
        lines[24] = "100                           -number of realizations\n"
        lines[35] = f"{np.max(ang1)} {np.max(ang2)} {np.max(ang3)}          -maximum search radii\n"
        lines[36] = (
            f"{np.max(a_hmax)*2} {np.max(a_hmin)*2} {np.max(a_hvert)*2}   "
            "-angles for search ellipsoid\n"
        )
        config_part = lines[:40]
        config_part.append(f"{nst}    {nugget}            -nst, nugget effect\n")
        for i in range(nst):
            config_part.append(
                f"{int(it[i])} {cc[i]} {ang1[i]} {ang2[i]} {ang3[i]}   -it,cc,ang1,ang2,ang3\n"
            )
            config_part.append(
                f"     {a_hmax[i]} {a_hmin[i]} {a_hvert[i]}         -a_hmax,a_hmin,a_vert\n"
            )
        with open(self.sisim_param, 'w') as f:
            f.writelines(config_part)

    def run_sisim(self) -> None:
        logging.info("Running SISIM simulation...")
        os.system(f"sisim.exe {self.sisim_param}")

    @staticmethod
    def load_sisim_output(filename: str = 'sisim.out') -> np.ndarray:
        with open(filename, 'r') as f:
            lines = f.readlines()
        grid_info = list(map(float, lines[1].split()))
        n_real = int(grid_info[-1])
        nx, ny, nz = int(grid_info[1]), int(grid_info[2]), int(grid_info[3])
        data_vals = np.array([float(line.split()[0]) for line in lines[3:-1]])
        if nz == 1:
            return data_vals.reshape((n_real, nx, ny))
        return data_vals.reshape((n_real, nx, ny, nz))

    def update_postsim(self, output_option: int, output_parameter: float) -> None:
        with open(self.postsim_param, 'r') as f:
            lines = f.readlines()
        lines[9] = f"{output_option}   {output_parameter}            -output option, output parameter\n"
        with open(self.postsim_param, 'w') as f:
            f.writelines(lines)

    def run_postsim(self) -> None:
        logging.info("Running PostSim...")
        os.system(f"postsim.exe {self.postsim_param}")

    @staticmethod
    def read_postsim_output(filename: str = 'postsim.out') -> pd.DataFrame:
        x_vals = np.repeat(np.arange(1, 51), 50)
        y_vals = np.tile(np.arange(1, 51), 50)
        df = pd.read_csv(filename, skiprows=4, sep=r'\s+', names=['Lower','Upper']).astype(float)
        df.insert(0, 'Yloc', y_vals)
        df.insert(0, 'Xloc', x_vals)
        return df

    @staticmethod
    def read_data_as_df(datafile: str, skip_header: int = 7) -> pd.DataFrame:
        return pd.read_csv(
            datafile, skiprows=skip_header, sep=r'\s+',
            names=['Xloc','Yloc','Primary','Secondary','Weight']
        ).astype(float)

    @staticmethod
    def compute_goodness_score(true_df: pd.DataFrame, post_df: pd.DataFrame) -> float:
        merged = pd.merge(
            true_df[['Xloc','Yloc','Primary']],
            post_df, on=['Xloc','Yloc'], how='inner'
        )
        within = (merged['Primary'] >= merged['Lower']) & (merged['Primary'] <= merged['Upper'])
        return np.mean(within)

    @staticmethod
    def integrate_area_difference(scores: np.ndarray, pvals: np.ndarray) -> float:
        return float(np.abs(np.trapz(scores - pvals, pvals)))

class QLearningOptimizer:
    def __init__(
        self,
        sisim_workflow: SISIMWorkflow,
        param_states: List[str],
        param_actions: List[str],
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.3,
        episodes: int = 10,
        steps_per_episode: int = 5
    ):
        self.workflow = sisim_workflow
        self.states = param_states
        self.actions = param_actions
        self.q_table = np.zeros((len(param_states), len(param_actions)))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode

    def generate_random_params(self) -> Dict[str, Any]:
        nst = np.random.randint(1, 4)
        cc = np.random.uniform(0.4, 0.99, size=nst)
        nugget = np.random.uniform(0.0, 0.4)
        ccsum = np.sum(cc) + nugget
        cc /= ccsum
        nugget /= ccsum
        it = np.random.randint(1, 3, size=nst)
        ang1 = np.random.randint(0, 45, size=nst)
        ang2 = np.zeros(nst)
        ang3 = np.zeros(nst)
        a_hmax = np.random.randint(1, 51, size=nst)
        a_hmin = np.array([random.randint(1, a_hmax[i]) for i in range(nst)])
        a_hvert = np.zeros(nst)
        return {
            'nst': nst, 'nugget': nugget, 'it': it, 'cc': cc,
            'ang1': ang1, 'ang2': ang2, 'ang3': ang3,
            'a_hmax': a_hmax, 'a_hmin': a_hmin, 'a_hvert': a_hvert
        }

    def adjust_param(self, params: Dict[str, Any], state: str, action: str) -> Dict[str, Any]:
        if state == 'nugget':
            params['nugget'] = min(params['nugget'] + 0.05, 0.4) if action == 'increase' else max(params['nugget'] - 0.05, 0.0)
            ssum = np.sum(params['cc']) + params['nugget']
            params['nugget'] /= ssum
            params['cc'] /= ssum
        elif state == 'cc':
            if action == 'increase':
                params['cc'] = np.clip(params['cc'] + 0.05, 0.01, 0.99)
            else:
                params['cc'] = np.clip(params['cc'] - 0.05, 0.01, 0.99)
            ssum = np.sum(params['cc']) + params['nugget']
            params['cc'] /= ssum
            params['nugget'] /= ssum
        elif state == 'ang1':
            if action == 'increase':
                params['ang1'] = np.clip(params['ang1'] + 5, 0, 90)
            else:
                params['ang1'] = np.clip(params['ang1'] - 5, 0, 90)
        return params

    def evaluate_fitness(self, param_dict: Dict[str, Any], df_true: pd.DataFrame, pvals: np.ndarray) -> float:
        self.workflow.update_sisim_variogram(**param_dict)
        self.workflow.run_sisim()
        scores = []
        for p in pvals:
            self.workflow.update_postsim(4, p)
            self.workflow.run_postsim()
            df_post = self.workflow.read_postsim_output()
            score = self.workflow.compute_goodness_score(df_true, df_post)
            scores.append(score)
        return self.workflow.integrate_area_difference(np.array(scores), pvals)

    def run_q_learning(self, df_true: pd.DataFrame, pvals: np.ndarray) -> Dict[str, Any]:
        best_reward = float('-inf')
        best_params: Optional[Dict[str, Any]] = None

        for _ in range(self.episodes):
            state = random.choice(self.states)
            current_params = self.generate_random_params()

            for _ in range(self.steps_per_episode):
                if random.random() < self.epsilon:
                    action = random.choice(self.actions)
                else:
                    s_idx = self.states.index(state)
                    a_idx = np.argmax(self.q_table[s_idx])
                    action = self.actions[a_idx]

                current_params = self.adjust_param(current_params, state, action)
                fit_val = self.evaluate_fitness(current_params, df_true, pvals)
                reward = 1.0 / (fit_val + 1e-9)
                s_idx = self.states.index(state)
                a_idx = self.actions.index(action)
                next_state = random.choice(self.states)
                ns_idx = self.states.index(next_state)
                self.q_table[s_idx, a_idx] += self.alpha * (
                    reward + self.gamma * np.max(self.q_table[ns_idx]) - self.q_table[s_idx, a_idx]
                )
                state = next_state
                if reward > best_reward:
                    best_reward = reward
                    best_params = current_params.copy()

        logging.info(f"Q-learning best reward = {best_reward:.4f}")
        return best_params if best_params is not None else {}

class GeneticAlgorithmOptimizer:
    def __init__(
        self,
        sisim_workflow: SISIMWorkflow,
        population_size: int = 50,
        generations: int = 10,
        crossover_prob: float = 0.5,
        mutation_prob: float = 0.1
    ):
        self.workflow = sisim_workflow
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

    def initialize_population(self) -> List[Dict[str, Any]]:
        return [self._random_variogram_param() for _ in range(self.population_size)]

    def _random_variogram_param(self) -> Dict[str, Any]:
        nst = np.random.randint(1, 4)
        cc = np.random.uniform(0.4, 0.99, size=nst)
        nugget = np.random.uniform(0.0, 0.4)
        ccsum = np.sum(cc) + nugget
        cc /= ccsum
        nugget /= ccsum
        it = np.random.randint(1, 3, size=nst)
        ang1 = np.random.randint(0, 90, size=nst)
        ang2 = np.random.randint(0, 90, size=nst)
        ang3 = np.random.randint(0, 90, size=nst)
        a_hmax = np.random.randint(1, 51, size=nst)
        a_hmin = np.array([random.randint(1, a_hmax[i]) for i in range(nst)])
        a_hvert = np.random.randint(1, 51, size=nst)
        return {
            'nst': nst, 'nugget': nugget, 'it': it, 'cc': cc,
            'ang1': ang1, 'ang2': ang2, 'ang3': ang3,
            'a_hmax': a_hmax, 'a_hmin': a_hmin, 'a_hvert': a_hvert
        }

    def evaluate_fitness(self, individual: Dict[str, Any], df_true: pd.DataFrame, pvals: np.ndarray) -> float:
        self.workflow.update_sisim_variogram(**individual)
        self.workflow.run_sisim()
        scores = []
        for p in pvals:
            self.workflow.update_postsim(4, p)
            self.workflow.run_postsim()
            df_post = self.workflow.read_postsim_output()
            score = self.workflow.compute_goodness_score(df_true, df_post)
            scores.append(score)
        return self.workflow.integrate_area_difference(np.array(scores), pvals)

    def tournament_selection(self, population: List[Dict[str, Any]], fits: List[float], t_size: int = 3) -> Dict[str, Any]:
        contenders = random.sample(list(zip(population, fits)), k=t_size)
        winner = min(contenders, key=lambda x: x[1])
        return winner[0]

    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.crossover_prob:
            child = {}
            child['nst'] = random.choice([parent1['nst'], parent2['nst']])
            child['nugget'] = (parent1['nugget'] + parent2['nugget']) / 2.0
            child['cc'] = random.choice([parent1['cc'], parent2['cc']]).copy()
            child['it'] = random.choice([parent1['it'], parent2['it']]).copy()
            child['ang1'] = random.choice([parent1['ang1'], parent2['ang1']]).copy()
            child['ang2'] = random.choice([parent1['ang2'], parent2['ang2']]).copy()
            child['ang3'] = random.choice([parent1['ang3'], parent2['ang3']]).copy()
            child['a_hmax'] = random.choice([parent1['a_hmax'], parent2['a_hmax']]).copy()
            child['a_hmin'] = random.choice([parent1['a_hmin'], parent2['a_hmin']]).copy()
            child['a_hvert'] = random.choice([parent1['a_hvert'], parent2['a_hvert']]).copy()
            total = child['nugget'] + np.sum(child['cc'])
            child['cc'] /= total
            child['nugget'] /= total
            return child
        return random.choice([parent1, parent2])

    def mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.mutation_prob:
            individual['nugget'] += random.uniform(-0.05, 0.05)
            individual['nugget'] = np.clip(individual['nugget'], 0.0, 0.4)
            ssum = np.sum(individual['cc']) + individual['nugget']
            individual['cc'] /= ssum
            individual['nugget'] /= ssum
        return individual

    def run_evolution(self, df_true: pd.DataFrame, pvals: np.ndarray) -> Dict[str, Any]:
        population = self.initialize_population()
        best_fitness = float('inf')
        best_solution: Optional[Dict[str, Any]] = None

        for g in range(self.generations):
            fitness_list = [self.evaluate_fitness(ind, df_true, pvals) for ind in population]
            idx_best = np.argmin(fitness_list)
            if fitness_list[idx_best] < best_fitness:
                best_fitness = fitness_list[idx_best]
                best_solution = population[idx_best].copy()
            logging.info(f"GA Gen {g+1}/{self.generations} Best Fitness: {best_fitness:.4f}")

            new_population = []
            while len(new_population) < self.population_size:
                p1 = self.tournament_selection(population, fitness_list, t_size=3)
                p2 = self.tournament_selection(population, fitness_list, t_size=3)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)
            population = new_population

        logging.info(f"GA best solution -> Fitness = {best_fitness:.4f}")
        return best_solution if best_solution is not None else {}

def main() -> None:
    workflow = SISIMWorkflow(sisim_param='sisim.par', postsim_param='postsim.par')
    workflow.split_data_file('cluster.dat', 'sample_data.txt', 'true_data.txt', frac=0.7)
    workflow.modify_sisim_datafile('sample_data.txt')

    df_true = workflow.read_data_as_df('true_data.txt', skip_header=7)
    p_intervals = np.linspace(0, 1, 21)

    logging.info("=== Q-LEARNING START ===")
    ql_opt = QLearningOptimizer(
        sisim_workflow=workflow,
        param_states=['nugget','cc','ang1'],
        param_actions=['increase','decrease'],
        alpha=0.1, gamma=0.9, epsilon=0.3,
        episodes=5, steps_per_episode=5
    )
    best_q_params = ql_opt.run_q_learning(df_true, p_intervals)
    logging.info(f"Q-Learning best parameters: {best_q_params}")

    logging.info("=== GENETIC ALGORITHM START ===")
    ga_opt = GeneticAlgorithmOptimizer(
        sisim_workflow=workflow,
        population_size=40,
        generations=5,
        crossover_prob=0.5,
        mutation_prob=0.2
    )
    best_ga_params = ga_opt.run_evolution(df_true, p_intervals)
    logging.info(f"GA best parameters: {best_ga_params}")

if __name__ == "__main__":
    main()
