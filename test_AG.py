# test_AG.py

import streamlit as st
from taller2ag import genetic_algorithm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
st.title("📐 Optimización GA: Distancia entre Paneles Solares")

# --- Entradas obligatorias ---
max_gens = st.number_input("Número de generaciones", min_value=1, value=50, step=1, help="El numero de generaciónes que el algoritmo genético realizará.")
latitude = st.number_input("Latitud (°)", min_value=-90.0, max_value=90.0, value=41.0, step=0.1, help="Latitud de la ubicación donde se instalarán los paneles solares.")
season = st.selectbox("Temporada", ["Invierno", "Verano"], help="Selecciona la temporada del año para la que deseas calcular la distancia mínima entre paneles solares.")

# --- Parámetros avanzados en expander ---
with st.expander("⚙️ Parámetros avanzados"):
    len_pop       = st.number_input("Tamaño población", min_value=2, value=10, step=1, help="Número de individuos en la población.")
    val_min       = st.number_input("Valor mínimo individuo (mm)", value=0.0, step=1.0, help="Valor mínimo que puede tomar un individuo, debe ser menor que el valor máximo.")
    val_max       = st.number_input("Valor máximo individuo (mm)", value=10000.0, step=1.0, help="Valor máximo que puede tomar un individuo, debe ser mayor que el valor mínimo.")
    dims          = st.text_input("Dimensiones panel B,L,H (mm)", "1476,659,35", help="Dimensiones del panel solar en mm. Separar por comas. Ejemplo: 1476,659,35 para un panel de 1476 mm de ancho, 659 mm de largo y 35 mm de alto.")
    incl_deg      = st.number_input("Inclinación β (°)", min_value=0.0, max_value=90.0, value=45.0, step=0.1, help="Inclinación del panel solar en grados. Debe ser entre 0 y 90 grados.")
    pm_min        = st.number_input("pm_min", min_value=0.0, max_value=1.0, value=0.1, step=0.01, help="Probabilidad mínima de mutación en el algoritmo genético.")
    pm_max        = st.number_input("pm_max", min_value=0.0, max_value=1.0, value=0.4, step=0.01, help="Probabilidad máxima de mutación en el algoritmo genético.")
    pc_min        = st.number_input("pc_min", min_value=0.0, max_value=1.0, value=0.3, step=0.01, help="Probabilidad mínima de cruce en el algoritmo genético.")
    pc_max        = st.number_input("pc_max", min_value=0.0, max_value=1.0, value=0.7, step=0.01, help="Probabilidad máxima de cruce en el algoritmo genético.")
    k_m           = st.number_input("k_m", min_value=0.0, value=0.035, step=0.005, help="Parámetro de mutación (k_m) para el algoritmo genético, controla la velocidad de cambio en el valor de la probabilidad de mutación cuando se usa el método de adaptación por diversidad.")
    k_c           = st.number_input("k_c", min_value=0.0, value=0.08, step=0.005, help="Parámetro de cruce (k_c) para el algoritmo genético, controla la velocidad de cambio en el valor de la probabilidad de cruce cuando se usa el método de adaptación por diversidad.")
    cross_method  = st.selectbox("Método de cruce", ["Cruce aritmetico","blx_alpha","sbx"], help="Selecciona el método de cruce que deseas utilizar en el algoritmo genético.")
    mut_method    = st.selectbox("Método de mutación", ["Mutación Gaussiana","Mutación Polinómica"], help="Selecciona el método de mutación que deseas utilizar en el algoritmo genético.")
    adaptation_method    = st.selectbox("Método de adaptación", ["Por Diversidad","Por Fitness"], help="Selecciona el método de adaptación que deseas utilizar en el algoritmo genético.")
    cross_param   = st.number_input("Parámetro cruce (α o η)", value=0.3, step=0.1, help="Parámetro de cruce (α o η) para el método de cruce seleccionado. Para el método aritmetico, este parámetro no es necesario y se puede dejar en 0, pero se recomienda usar un valor entre 0 y 1. Para el método BLX, este parámetro debe estar entre 0 y 1. Para el método SBX, este parámetro debe estar entre 2 y 5.")
    mut_param     = st.number_input("Parámetro mutación (σ o ηₘ)", value=500.0, step=10.0, help="Parámetro de mutación (σ o ηₘ) para el método de mutación seleccionado. Para el método de mutación polinómica, este parámetro debe estar entre 0 y 1. Para el método de mutación gaussiana, este parámetro debe estar entre 0 y 1000.")

# --- Botón de ejecución ---
if st.button("▶️ Ejecutar GA"):
    B,L,H = [float(x) for x in dims.split(",")]
    panel_dims = [B,L,H]
    
    
    # Cambiar nombres de métodos de cruce, mutación y adaptación a los nombres usados en el código
    if adaptation_method == "Por Diversidad":
        adaptation_method = "diversity"
    elif adaptation_method == "Por Fitness":   
        adaptation_method = "fitness"
        
    if cross_method == "Cruce aritmetico":
        cross_method = "arithmetic_crossover"
    elif cross_method == "blx_alpha":
        cross_method = "blx_alpha"
    elif cross_method == "sbx":
        cross_method = "sbx"
        
    if mut_method == "Mutación Gaussiana":
        mut_method = "gaussian_mutation"
    elif mut_method == "Mutación Polinómica":
        mut_method = "polynomial_mutation"
    
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
    
    # verificar que las dimensiones no sean ninguna menor o igual a 0.0
    if any(d <= 0.0 for d in panel_dims):
        st.error("❌ Dimensiones del panel deben ser mayores a 0.0 mm")
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
    # Explicación de la grafica
    st.markdown("### Evolución de las probabilidades de cruce y mutación")
    st.markdown("La gráfica muestra la evolución de las probabilidades de cruce $(p_c)$ y mutación $(p_m)$ a lo largo de las generaciones. "
                "La línea azul claro representa la probabilidad de cruce $(p_c)$ y la línea azul oscuro representa la probabilidad de mutación $(p_m)$. "
                "Ambas probabilidades se adaptan dinámicamente en función del método de adaptación seleccionado para permitir equilibrio entre la exploración y explotación según sea necesario.")

    df = pd.DataFrame(result["history"], columns=["pm","pc"])
    st.line_chart(df)

    # 2) Gráfica de evolución de los mejores
    # Explicación de la grafica
    st.markdown("### Evolución de los mejores individuos")
    st.markdown("La gráfica muestra la evolución de los mejores individuos a lo largo de las generaciones. "
                "La línea azul claro representa el mejor individuo en cada generación, que corresponde a la distancia entre paneles solares. "
                "A medida que avanza el algoritmo genético, se espera que la distancia entre paneles solares converja hacia un valor óptimo. ")

    df_bests = pd.DataFrame({"generación": range(1, len(result["bests"])+1),
                             "mejor": result["bests"]})
    st.line_chart(df_bests.set_index("generación"))

    # 3) Gráfica de paneles solares
    # Explicación de la gráfica
    st.markdown("### Esquema de paneles solares y distancia ideal")
    st.markdown("El esquema muestra dos paneles solares instalados en una inclinación de β grados. "
                "La distancia ideal entre los paneles solares está indicada por la línea de color gris horizontal discontinua. "
                "La trayectoria del rayo solar desde la punta del primer panel hasta la base del segundo panel está representada por la línea verde discontinua. "
                
                "Este esquema ilustra la relación entre la inclinación de los paneles, la distancia ideal y la trayectoria solar.")

    # Parámetros para la gráfica del panel solar:
    B = panel_dims[0]        # longitud del panel
    incl = incl_deg          # inclinación en grados
    ideal = result["target"] # distancia ideal entre paneles

    # Coordenadas del primer panel (origen)
    x0, y0 = 0, 0
    x1 = B * np.cos(np.deg2rad(incl))
    y1 = B * np.sin(np.deg2rad(incl))

    # Coordenadas del segundo panel (desplazado en x por "ideal")
    dx = ideal
    x2 = dx
    y2 = 0
    x3 = x2 + B * np.cos(np.deg2rad(incl))
    y3 = B * np.sin(np.deg2rad(incl))

    fig, ax = plt.subplots()

    # Dibuja paneles
    ax.plot([x0, x1], [y0, y1], color='tab:blue', linewidth=5)  # panel izquierdo
    ax.plot([x2, x3], [y2, y3], color='tab:orange', linewidth=5)  # panel derecho

    # Línea horizontal indicando distancia ideal
    ax.hlines(0, x1, x2, colors='gray', linestyles='dashed')
    ax.text((x1 + x2)/2, -B*0.05, f"{ideal:.1f} mm", ha='center', va='top', fontsize=10)

    # Ángulo de inclinación como arco
    angle_arc = patches.Arc((0, 0), width=B*0.6, height=B*0.6, angle=0,
                            theta1=0, theta2=incl, color='black', linestyle='-')
    ax.add_patch(angle_arc)

    # Texto del ángulo (ubicado ligeramente dentro del arco)
    angle_label_radius = B * 0.35
    angle_label_x = angle_label_radius * np.cos(np.deg2rad(incl / 2))
    angle_label_y = angle_label_radius * np.sin(np.deg2rad(incl / 2))
    ax.text(angle_label_x, angle_label_y*-0.2, f"β = {incl}°", fontsize=10, ha='center', va='center')
    
    # Rayo solar desde punta del primer panel a base del segundo
    ax.plot([x1, x2], [y1, y2], linestyle='dashdot', color='green')
    ax.text((x1 + x2)/2, (y1 + y2)/2 + B*0.01, "Rayo solar", 
            rotation=np.degrees(np.arctan2(y2 - y1, x2 - x1)),
            va='bottom', ha='center', fontsize=9)

    # Ajustes de la gráfica
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-B * 0.2, x3 + B * 0.2)
    ax.set_ylim(-B * 0.2, max(y1, y3) + B * 0.2)
    ax.set_xlabel("Distancia horizontal (mm)", labelpad=10)
    ax.set_ylabel("Altura (mm)", labelpad=10)
    ax.set_title("Esquema: Paneles solares, distancia ideal y trayectoria solar")

    st.pyplot(fig)

    # Mostrar población y mejor por generación
    for i, (pop, best, history) in enumerate(zip(result["populations"], result["bests"], result["history"])):
        st.markdown(f"#### Generación {i+1} ")
        st.markdown(f"$p_c =$ {round(history[1],4)}")
        st.markdown(f"$p_m =$ {round(history[0],4)}")
        st.write("Población:", [round(d,2) for d in pop])
        st.markdown(f"##### Mejor individuo: `{round(best,2)}`")
