import streamlit as st
from pathlib import Path


st.set_page_config(
        page_title="Ground-based Reflected Light Imaging Planner",
        page_icon="images/grip-offwhite.png",
        layout="wide",
    )

sidebar_logo = 'images/grip-offwhite.png'
st.logo(sidebar_logo, size='large')

st.sidebar.write("""<div style="width:100%;text-align:center;"><a href="https://docs.google.com/spreadsheets/d/1sk9wDIVi4uWELL_gSyr7Zb41HZJ27Puyvc1QqWBZFs4/edit?gid=0#gid=0" style="text-decoration:none; color:#EEEEEE">Indirectly Discovered, \n Directly Detected Companions</a></div>""", unsafe_allow_html=True)

# left_co, cent_co,last_co = st.columns(3)
# with cent_co:
#     st.image('images/logo.png', width=300)

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

with open('target-list-compile.tar.gz', 'rb') as f:
    st.download_button('Download this notebook and files',f)

intro_markdown = read_markdown_file("TargetList-CompileGroundBasedReflLightTargets.md")
st.markdown(intro_markdown, unsafe_allow_html=True)
