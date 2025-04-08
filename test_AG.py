# test_AG.py

import streamlit as st
from taller2ag import genetic_algorithm
import pandas as pd
import numpy as np
st.title("📐 Optimización GA: Distancia entre Paneles Solares")

# --- Entradas obligatorias ---
max_gens = st.number_input("Número de generaciones", min_value=1, value=50, step=1)
latitude = st.number_input("Latitud (°)", min_value=-90.0, max_value=90.0, value=41.0, step=0.1)
season = st.selectbox("Temporada", ["Invierno", "Verano"])

# --- Parámetros avanzados en expander ---
with st.expander("⚙️ Parámetros avanzados"):
    len_pop       = st.number_input("Tamaño población", min_value=2, value=10, step=1)
    val_min       = st.number_input("Valor mínimo individuo (mm)", value=0.0, step=1.0)
    val_max       = st.number_input("Valor máximo individuo (mm)", value=10000.0, step=1.0)
    dims          = st.text_input("Dimensiones panel B,L,H (mm)", "1476,659,35")
    incl_deg      = st.number_input("Inclinación β (°)", min_value=0.0, max_value=90.0, value=45.0, step=0.1)
    pm_min        = st.number_input("pm_min", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    pm_max        = st.number_input("pm_max", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
    pc_min        = st.number_input("pc_min", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    pc_max        = st.number_input("pc_max", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    k_m           = st.number_input("k_m", min_value=0.0, value=0.035, step=0.005)
    k_c           = st.number_input("k_c", min_value=0.0, value=0.08, step=0.005)
    cross_method  = st.selectbox("Método de cruce", ["arithmetic_crossover","blx_alpha","sbx"])
    mut_method    = st.selectbox("Método de mutación", ["gaussian_mutation","polynomial_mutation"])
    cross_param   = st.number_input("Parámetro cruce (α o η)", value=0.3, step=0.1)
    mut_param     = st.number_input("Parámetro mutación (σ o ηₘ)", value=500.0, step=10.0)

# --- Botón de ejecución ---
if st.button("▶️ Ejecutar GA"):
    B,L,H = [float(x) for x in dims.split(",")]
    panel_dims = [B,L,H]

    # Volver none cross_param = 0 si el método es arithmetic_crossover
    if cross_method == "arithmetic_crossover" and cross_param <= 0.0:
        cross_param = None
    # verificar que cross_param no sea mayor a 5 ni menor a 2 si el método es sbx
    if cross_method == "sbx" and (cross_param < 2.0 or cross_param > 5.0):
        st.error("❌ Parámetro de cruce (α) debe ser entre 2 y 5 para el método SBX")
        st.stop()
    # verificar que cross_param no sea menor a 0.0 ni mayor a 1 si el método es blx_alpha o arithmetic_crossover
    if cross_method == "blx_alpha" and (cross_param < 0.0 or cross_param > 1.0):
        st.error("❌ Parámetro de cruce (α) debe ser entre 0 y 1 para el método BLX o arithmetic_crossover")
        st.stop()
    # verificar que mut_param no sea menor a 20 ni mayor a 100 si el método es polynomial_mutation
    if mut_method == "polynomial_mutation" and (mut_param < 0.0 or mut_param > 1.0):
        st.error("❌ Parámetro de mutación (ηₘ) debe ser entre 0 y 1 para el método polynomial_mutation")
        st.stop()
    if mut_method == "gaussian_mutation" and (mut_param < 0.0 or mut_param > 1000.0):
        st.error("❌ Parámetro de mutación (σ) debe ser entre 0 y 1000 para el método gaussian_mutation")
        st.stop()

    result = genetic_algorithm(
        max_generations   = max_gens,
        latitude          = latitude,
        season            = season.lower(),
        len_poblation     = len_pop,
        valor_min         = val_min,
        valor_max         = val_max,
        panel_dimensions  = panel_dims,
        inclination_degree= incl_deg,
        pm_min            = pm_min,
        pm_max            = pm_max,
        pc_min            = pc_min,
        pc_max            = pc_max,
        k_m               = k_m,
        k_c               = k_c,
        crossover_method  = cross_method,
        mutation_method   = mut_method,
        crossover_param   = cross_param,
        mutation_param    = mut_param
    )

    st.success("✅ Ejecución completada")
    st.write("**Distancia Ideal (mm):**", round(result["target"],2))
    st.write("**Mejor individuo (distancia mm):**", round(result["best"],2))
    st.write("**Error (mm):**", round(result["fitness"],2))

    # Gráfica de evolución de pm y pc
    df = pd.DataFrame(result["history"], columns=["pm","pc"])
    st.line_chart(df)

    # Mostrar población y mejor por generación
    for i, (pop, best, history) in enumerate(zip(result["populations"], result["bests"], result["history"])):
        st.markdown(f"#### Generación {i+1} ")
        st.markdown(f"$p_c =$ {round(history[0],4)}")
        st.markdown(f"$p_c =$ {round(history[1],4)}")
        st.write("Población:", [round(d,2) for d in pop])
        st.markdown(f"##### Mejor individuo: `{round(best,2)}`")
