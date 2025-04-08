import streamlit as st

st.title("Taller Segundo Momento Algoritmos Genéticos")

st. markdown("""
            ## Integrantes:
            ### John Sebastian Galindo Hernandez
            ### Miguel Ángel Moreno Beltrán
            
            ### Fecha: 08 de Abril de 2025
            
            ### Objetivo:
            Un ingeniero de sistemas y computación ha sido encargado de desarrollar una solución que 
            optimice la instalación de paneles solares fotovoltaicos del tipo A-135P, cuyas dimensiones son 
            1476 mm × 659 mm × 35 mm. Estos paneles serán instalados en una ubicación con una latitud 
            de 41 grados y deberán ser inclinados 45 grados para maximizar su eficiencia energética a lo 
            largo del año. 
            Problema Para Resolver 
            Es necesario calcular la distancia mínima entre paneles solares para evitar sombras y 
            maximizar la captación de energía en diferentes condiciones estacionales: 
            1. Para todo el año, considerando la incidencia solar en invierno como caso crítico. 
            2. Para la temporada de verano, donde las condiciones permiten una posible reducción 
            de la distancia entre paneles. 
            
            ### Requisitos:
            Requisitos de la Solución 
            El ingeniero deberá diseñar e implementar un algoritmo genético adaptativo (AGA) que 
            optimice la distancia mínima entre los paneles solares. La solución debe incluir: 
            • Un modelo de optimización basado en algoritmos genéticos, con ajuste dinámico de 
            los parámetros de cruce y mutación. 
            • Interfaz interactiva, que permita a los usuarios ingresar la latitud y seleccionar la 
            temporada del año (invierno o verano). 
            • Un frontend didáctico e intuitivo, que facilite la configuración de los parámetros y la 
            visualización de los resultados. 
            
            ### Criterios de Evaluación:
            Criterios de Evaluación 
            • Precisión del cálculo de distancia óptima en función de la latitud y temporada. 
            • Correcta implementación del algoritmo genético adaptativo, con parámetros 
            ajustables. 
            • Interfaz funcional e interactiva, con opciones claras para el usuario. 
            • Documentación clara del código y explicación del modelo utilizado. 
            """)

st.write("Link Del Documento:")
st.write("Link Del Repositorio: https://github.com/SebasGalindo/AGSegundoMomento")
