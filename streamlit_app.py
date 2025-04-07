import streamlit as st

pages = {
    "Principal": [
        st.Page("homepage.py", title="Taller Segundo Momento AG"),
        st.Page("test_AG.py", title="Prueba del AGA"),
    ],
    "Explicaciones": [
        st.Page("explanation.py", title="Explicación del Algoritmo Genético Adaptativo"),
    ],
}

pg = st.navigation(pages)
pg.run()

