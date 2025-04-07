import streamlit as st

pages = {
    "Principal": [
        st.Page("homepage.py", title="Tomate ResNet"),
        st.Page("test_AG.py", title="Prueba de Clasificación"),
    ],
    "Explicaciones": [
        st.Page("explanation.py", title="Explicación del Dataset"),
    ],
}

pg = st.navigation(pages)
pg.run()

