import streamlit as st

pg = st.navigation(
    [st.Page("homepage.py", title="Welcome!", icon=":material/add_circle:"),
     st.Page("machine_learning_app.py", title="Machine Learning Processor", icon=":material/public:")
     ]
     )

pg.run()