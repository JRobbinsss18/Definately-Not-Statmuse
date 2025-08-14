import streamlit as st
from src.dashboard import NBASportsMuseDashboard

def main():
    st.set_page_config(
        page_title="Definately NOT SportsMuse",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    dashboard = NBASportsMuseDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()