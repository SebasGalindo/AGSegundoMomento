import random
import math
import statistics

# --- Funciones auxiliares ---

def create_individual(valor_min, valor_max):
    return random.uniform(valor_min, valor_max)

def create_poblation(len_poblation, valor_min, valor_max):
    return [create_individual(valor_min, valor_max) for _ in range(len_poblation)]

def min_angle(latitude, solar_decline):
    return 90 - latitude + solar_decline

def min_distance(panel_dimensions, latitude, inclination_degree, solar_decline):
    B = panel_dimensions[0]
    beta = math.radians(inclination_degree)
    alpha_min = math.radians(min_angle(latitude, solar_decline))
    return B * math.cos(beta) + (B * math.sin(beta)) / math.tan(alpha_min)

def fitness(x, target_distance):
    return abs(x - target_distance)

def ranking_selection(poblation, target_distance, sub_poblation_percentage):
    len_sub = int(len(poblation) * sub_poblation_percentage)
    idxs = random.sample(range(len(poblation)), len_sub)
    sub = [poblation[i] for i in idxs]
    return min(sub, key=lambda x: fitness(x, target_distance))

# --- Cruces y mutaciones ---

def arithmetic_crossover(p1, p2, alpha=None):
    if alpha is None: alpha = random.random()
    return (alpha*p1 + (1-alpha)*p2,
            (1-alpha)*p1 + alpha*p2)

def blx_alpha(p1, p2, alpha=0.3):
    lo, hi = min(p1,p2), max(p1,p2)
    d = hi - lo
    return (random.uniform(lo - alpha*d, hi + alpha*d),
            random.uniform(lo - alpha*d, hi + alpha*d))

def sbx(p1, p2, eta=2):
    u = random.random()
    if u <= 0.5:
        beta = (2*u)**(1/(eta+1))
    else:
        beta = (1/(2*(1-u)))**(1/(eta+1))
    return (0.5*((1+beta)*p1 + (1-beta)*p2),
            0.5*((1-beta)*p1 + (1+beta)*p2))

def gaussian_mutation(D, D_min, D_max, sigma=500):
    Dp = D + random.gauss(0, sigma)
    return min(max(Dp, D_min), D_max)

def polynomial_mutation(D, D_min, D_max, eta_m=20):
    u = random.random()
    if u < 0.5:
        delta = (2*u)**(1/(1+eta_m)) - 1
    else:
        delta = 1 - (2*(1-u))**(1/(1+eta_m))
    if delta < 0:
        Dp = D + delta*(D - D_min)
    else:
        Dp = D + delta*(D_max - D)
    return min(max(Dp, D_min), D_max)

# --- Métodos de adaptación ---

def diversity_adaptation(poblation, pm_min, pm_max, pc_min, pc_max, k_m, k_c):
    D = statistics.pstdev(poblation)
    pm = pm_min + (pm_max - pm_min)*math.exp(-k_m * D)
    pc = pc_max - (pc_max - pc_min)*math.exp(-k_c * D)
    return pm, pc

def avg_fitness(poblation, target_distance):
    return sum(fitness(d, target_distance) for d in poblation) / len(poblation)

# --- Función principal GA ---

def genetic_algorithm(
    max_generations:int,
    latitude:float,
    season:str,
    # parámetros de población y distancia
    len_poblation:int=10,
    valor_min:float=0,
    valor_max:float=10000,
    panel_dimensions:list=[1476,659,35],
    inclination_degree:float=45,
    winter_solar_decline:float=-23.45,
    summer_solar_decline:float=23.45,
    # adaptación
    diversity_method:str="diversity",      # "diversity" o "fitness"
    pm_min:float=0.1, pm_max:float=0.4,
    pc_min:float=0.3, pc_max:float=0.7,
    k_m:float=0.035, k_c:float=0.08,
    # operadores
    crossover_method:str="sbx",
    mutation_method:str="gaussian_mutation",
    crossover_param:float=None,
    mutation_param:float=None
    ) -> dict:
    """
    Devuelve un dict con:
      'best': mejor individuo,
      'fitness': fitness(mejor, target_distance),
      'history': lista de tuplas (pm, pc) por generación,
      'populations': lista de poblaciones por generación,
      'bests': lista de mejores por generación
    """
    sd = summer_solar_decline if season.lower() in ["verano","summer"] else winter_solar_decline
    target = min_distance(panel_dimensions, latitude, inclination_degree, sd)

    poblacion = create_poblation(len_poblation, valor_min, valor_max)
    fit_prev = avg_fitness(poblacion, target) if diversity_method=="fitness" else None

    history = []
    pop_history = []
    best_history = []
    for gen in range(max_generations):
        pop_history.append(poblacion.copy())
        best_curr = ranking_selection(poblacion, target, 1.0)
        best_history.append(best_curr)
        if diversity_method=="diversity":
            pm, pc = diversity_adaptation(poblacion, pm_min, pm_max, pc_min, pc_max, k_m, k_c)
        else:
            fit_curr = avg_fitness(poblacion, target)
            deltaF = (fit_prev - fit_curr)/fit_prev if fit_prev else 0
            fit_prev = fit_curr
            pm = pm_min + (pm_max - pm_min)*(1 - deltaF)
            pc = pc_min + (pc_max - pc_min)*deltaF
        history.append((pm, pc))

        nueva = []
        while len(nueva) < len_poblation:
            p1 = ranking_selection(poblacion, target, 0.5)
            p2 = ranking_selection(poblacion, target, 0.5)
            if random.random() < pc:
                fn = globals()[crossover_method]
                h1,h2 = fn(p1, p2) if crossover_param is None else fn(p1, p2, crossover_param)
            else:
                h1,h2 = p1,p2
            fnm = globals()[mutation_method]
            if random.random() < pm:
                args = (h1, valor_min, valor_max) if mutation_method=="gaussian_mutation" else (h1, valor_min, valor_max, mutation_param or 20)
                h1 = fnm(*args)
            if random.random() < pm:
                args = (h2, valor_min, valor_max) if mutation_method=="gaussian_mutation" else (h2, valor_min, valor_max, mutation_param or 20)
                h2 = fnm(*args)
            nueva.extend([h1,h2])
        poblacion = nueva[:len_poblation]

    pop_history.append(poblacion.copy())
    best_history.append(ranking_selection(poblacion, target, 1.0))

    best = best_history[-1]
    return {
        "target": target,
        "best": best,
        "fitness": fitness(best, target),
        "history": history,
        "populations": pop_history,
        "bests": best_history
    }
