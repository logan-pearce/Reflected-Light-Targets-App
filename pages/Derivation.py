import streamlit as st
from pathlib import Path


st.set_page_config(
        page_title="Reflected Light Planets",
        page_icon="images/star-only-orange-transp.png",
        layout="wide",
    )

sidebar_logo = 'images/star-only-orange-transp.png'
st.logo(sidebar_logo, size='large')

# left_co, cent_co,last_co = st.columns(3)
# with cent_co:
#     st.image('images/logo.png', width=300)

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

with open('target-list-compile.tar.gz', 'rb') as f:
    st.download_button('Download this notebook and files',f)

intro_markdown = read_markdown_file("TargetList-CompileGroundBasedReflLightTargets.md")
st.markdown(intro_markdown, unsafe_allow_html=True)
