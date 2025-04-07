import streamlit as st

st.title("📝 Explicación: Algoritmo Genético Adaptativo para Distancia entre Paneles Solares")

st.markdown("""
**Objetivo base**: Optimizar la distancia mínima entre paneles solares fotovoltaicos del tipo A-135P (1476×659×35 mm) instalados a una latitud de 41°, inclinados 45°, para evitar sombras en invierno y reducir distancia en verano.

Este problema se plantea en el PDF del Taller AGA, donde se define:

- Cálculo de la elevación solar crítica:
  - Invierno: $\\alpha_{inv}=(90^\circ-\\varphi)-23.45^\circ$
  - Verano:  $\\alpha_{ver}=(90^\circ-\\varphi)+23.45^\circ$
- Fórmula de distancia mínima:
  $D_M = B\cos\\beta + \\frac{B\sin\\beta}{\\tan\\alpha}$
  
  Donde:
  - **B**: longitud del panel (1476 mm)
  - **β**: ángulo de inclinación (45°)
  - **α**: elevación solar crítica (invierno/verano)
""", unsafe_allow_html=True)

st.header("1. Representación y Fitness")
st.markdown("""
- **Cromosoma**: un valor real `D` (distancia en mm).
- **Fitness**: mide la cercanía al valor óptimo teórico:

```python
def fitness(D, D_opt):
    return abs(D - D_opt)
```
""", unsafe_allow_html=True)

st.header("2. Selección por Torneo")
st.markdown("Se eligen aleatoriamente un subconjunto de la población y se selecciona el individuo con menor fitness:")
st.code("""def ranking_selection(pob, D_opt, pct):
    import random
    # toma pct% de la población
    sub = random.sample(pob, int(len(pob)*pct))
    return min(sub, key=lambda x: abs(x - D_opt))
""", language='python')

st.header("3. Cruce y Mutación")
st.subheader("3.1 Cruce Aritmético")
st.code("""def arithmetic_crossover(p1, p2, alpha=None):
    import random
    if alpha is None:
        alpha = random.random()
    return (alpha*p1 + (1-alpha)*p2,
            (1-alpha)*p1 + alpha*p2)
""", language='python')

st.subheader("3.2 BLX-α")
st.code("""def blx_alpha(p1, p2, alpha=0.3):
    import random
    lo, hi = min(p1,p2), max(p1,p2)
    d = hi - lo
    return (random.uniform(lo - alpha*d, hi + alpha*d),
            random.uniform(lo - alpha*d, hi + alpha*d))
""", language='python')

st.subheader("3.3 SBX (Simulated Binary Crossover)")
st.code("""def sbx(p1, p2, eta=2):
    import random
    u = random.random()
    if u <= 0.5:
        beta = (2*u)**(1/(eta+1))
    else:
        beta = (1/(2*(1-u)))**(1/(eta+1))
    return (0.5*((1+beta)*p1 + (1-beta)*p2),
            0.5*((1-beta)*p1 + (1+beta)*p2))
""", language='python')

st.subheader("3.4 Mutación Gaussiana")
st.code("""def gaussian_mutation(D, D_min, D_max, sigma=500):
    import random
    Dp = D + random.gauss(0, sigma)
    return min(max(Dp, D_min), D_max)
""", language='python')

st.subheader("3.5 Mutación Polinómica")
st.code("""def polynomial_mutation(D, D_min, D_max, eta_m=20):
    import random
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
""", language='python')

st.header("4. Adaptación de Parámetros")
st.markdown("""
Existen dos métodos principales:

1. **Basado en diversidad**:
   ```python
   import statistics, math
   D = statistics.pstdev(poblacion)
   pm = pm_min + (pm_max-pm_min)*math.exp(-k_m*D)
   pc = pc_max - (pc_max-pc_min)*math.exp(-k_c*D)
   ```

2. **Basado en tasa de mejora**:
   ```python
   fit_prev = avg_fitness(poblacion, D_opt)
   fit_curr = avg_fitness(nueva_pob, D_opt)
   deltaF = (fit_prev - fit_curr)/fit_prev
   pm = pm_min + (pm_max-pm_min)*(1 - deltaF)
   pc = pc_min + (pc_max-pc_min)*deltaF
   ```
""", unsafe_allow_html=True)

st.header("5. Integración con Streamlit")
st.markdown("A continuación se muestra cómo llamar al GA desde una app Streamlit y visualizar resultados:")
st.code("""from taller2ag import genetic_algorithm
import streamlit as st

# Parámetros de entrada
generations = st.number_input('Generaciones', 1, 1000, 50)
latitude    = st.number_input('Latitud', -90.0, 90.0, 41.0)
season      = st.selectbox('Temporada', ['Invierno','Verano'])

# Ejecución del GA
if st.button('Ejecutar GA'):
    result = genetic_algorithm(
        max_generations=generations,
        latitude=latitude,
        season=season.lower(),
        # ... parámetros avanzados ...
    )
    st.write('Mejor distancia (mm):', result['best'])
    st.line_chart(result['history'])
""", language='python')

st.markdown("""
**Conclusión**: Con esta estructura en modo notebook, puedes entender cada parte del AGA, desde la representación del cromosoma hasta la integración final en Streamlit. Ajusta parámetros y operadores para experimentar y optimizar la instalación de paneles solares en diferentes estaciones y latitudes.
""", unsafe_allow_html=True)
